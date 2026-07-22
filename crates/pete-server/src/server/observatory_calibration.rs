#[derive(Clone, Debug, Default, Deserialize)]
struct CalibrationConsoleQuery {
    snapshot_id: Option<String>,
    at_or_before_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDofView {
    pub name: String,
    pub observable: bool,
    pub covariance: Option<f32>,
    pub evidence_count: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConsumerGate {
    pub consumer: String,
    pub allowed: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationPlotSeries {
    pub metric: String,
    pub unit: Option<String>,
    pub points: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEstimatorView {
    pub id: String,
    pub label: String,
    pub configured_prior: serde_json::Value,
    pub live_estimate: serde_json::Value,
    pub trust_state: String,
    pub confidence: Option<f32>,
    pub uncertainty: serde_json::Value,
    pub degrees_of_freedom: Vec<CalibrationDofView>,
    pub evidence_counts: serde_json::Value,
    pub residuals: serde_json::Value,
    pub thresholds: serde_json::Value,
    pub rejection_reasons: Vec<String>,
    pub evidence_age_ms: Option<u64>,
    pub epoch: Option<String>,
    pub epoch_changed: bool,
    pub invalidation_reason: Option<String>,
    pub held_out_validation: serde_json::Value,
    pub consumers: Vec<CalibrationConsumerGate>,
    pub plots: Vec<CalibrationPlotSeries>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConsoleResponse {
    pub snapshot_id: String,
    pub at_ms: u64,
    pub estimators: Vec<CalibrationEstimatorView>,
    pub transition_event_ids: Vec<String>,
}

fn build_calibration_console(
    selection: &ObservatoryNowSelection,
    events: &[BrainEvent],
) -> CalibrationConsoleResponse {
    let now = serde_json::to_value(&selection.selected.now).unwrap_or_default();
    let previous = selection
        .previous
        .as_ref()
        .and_then(|entry| serde_json::to_value(&entry.now).ok());
    let mut estimators = Vec::new();
    estimators.push(calibration_view(
        "kinect_geometry",
        "Kinect geometry / mount",
        serde_json::to_value(pete_now::CalibrationStateConfig::default()).unwrap_or_default(),
        now.pointer("/kinect/live_geometry_calibration").cloned(),
        previous
            .as_ref()
            .and_then(|value| value.pointer("/kinect/live_geometry_calibration"))
            .cloned(),
        selection.selected.now.t_ms,
    ));
    estimators.push(calibration_view(
        "imu",
        "IMU bias, noise, and mounting",
        serde_json::to_value(pete_now::ImuCalibrationConfig::default()).unwrap_or_default(),
        now.pointer("/imu/calibration").cloned(),
        previous
            .as_ref()
            .and_then(|value| value.pointer("/imu/calibration"))
            .cloned(),
        selection.selected.now.t_ms,
    ));
    let latency = now
        .pointer("/extensions/sensor.latency_calibration")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let previous_latency = previous
        .as_ref()
        .and_then(|value| value.pointer("/extensions/sensor.latency_calibration"))
        .and_then(serde_json::Value::as_object);
    if latency.is_empty() {
        estimators.push(calibration_view(
            "timing",
            "Sensor timing / latency",
            serde_json::to_value(pete_now::LatencyCalibrationConfig::default()).unwrap_or_default(),
            None,
            None,
            selection.selected.now.t_ms,
        ));
    } else {
        for (stream, estimate) in latency {
            estimators.push(calibration_view(
                &format!("timing:{stream}"),
                &format!("Timing: {stream}"),
                serde_json::to_value(pete_now::LatencyCalibrationConfig::default())
                    .unwrap_or_default(),
                Some(estimate),
                previous_latency
                    .and_then(|values| values.get(&stream))
                    .cloned(),
                selection.selected.now.t_ms,
            ));
        }
    }
    estimators.push(calibration_view(
        "locomotion",
        "Locomotion / wheel calibration",
        serde_json::to_value(pete_now::LocomotionCalibrationConfig::default()).unwrap_or_default(),
        now.pointer("/extensions/calibration.locomotion").cloned(),
        previous
            .as_ref()
            .and_then(|value| value.pointer("/extensions/calibration.locomotion"))
            .cloned(),
        selection.selected.now.t_ms,
    ));
    for estimator in &mut estimators {
        attach_estimator_semantics(estimator);
    }
    CalibrationConsoleResponse {
        snapshot_id: selection.selected.snapshot_id.clone(),
        at_ms: selection.selected.now.t_ms,
        estimators,
        transition_event_ids: events
            .iter()
            .filter(|event| event.event_type == BrainEventType::CalibrationTransition)
            .map(|event| event.event_id.0.clone())
            .collect(),
    }
}

fn calibration_view(
    id: &str,
    label: &str,
    configured_prior: serde_json::Value,
    live: Option<serde_json::Value>,
    previous: Option<serde_json::Value>,
    now_ms: u64,
) -> CalibrationEstimatorView {
    let live_estimate = live.unwrap_or_else(|| serde_json::json!({"status":"not_observed"}));
    let trust_state = value_string(&live_estimate, &["trust_state"])
        .unwrap_or_else(|| "not_observed".to_string());
    let confidence = value_f32(&live_estimate, &["confidence", "clock_confidence"]);
    let epoch = epoch_string(&live_estimate);
    let previous_epoch = previous.as_ref().and_then(epoch_string);
    let updated_at_ms = value_u64(
        &live_estimate,
        &[
            "updated_at_ms",
            "last_observed_at_ms",
            "epoch_started_at_ms",
        ],
    );
    let invalidation_reason =
        value_string(&live_estimate, &["invalidation_reason"]).or_else(|| {
            live_estimate
                .pointer("/epoch/invalidation_reason")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        });
    let thresholds = configured_thresholds(&configured_prior);
    let mut notes = Vec::new();
    if let (Some(previous_conditions), Some(current_conditions)) = (
        previous.as_ref().and_then(|value| value.get("conditions")),
        live_estimate.get("conditions"),
    ) {
        if previous_conditions != current_conditions {
            notes.push(format!(
                "surface/tire conditions changed: {previous_conditions} -> {current_conditions}"
            ));
        }
    }
    CalibrationEstimatorView {
        id: id.to_string(),
        label: label.to_string(),
        configured_prior,
        uncertainty: live_estimate
            .get("covariance")
            .cloned()
            .or_else(|| live_estimate.get("uncertainty").cloned())
            .or_else(|| live_estimate.get("transport_latency").cloned())
            .unwrap_or(serde_json::Value::Null),
        degrees_of_freedom: calibration_dofs(id, &live_estimate),
        evidence_counts: live_estimate
            .get("evidence_counts")
            .cloned()
            .unwrap_or_else(|| evidence_summary(&live_estimate)),
        residuals: live_estimate
            .get("residuals")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        thresholds,
        rejection_reasons: string_array(&live_estimate, "rejection_reasons"),
        evidence_age_ms: updated_at_ms.map(|updated| now_ms.saturating_sub(updated)),
        epoch_changed: previous_epoch.is_some() && previous_epoch != epoch,
        epoch,
        invalidation_reason,
        held_out_validation: live_estimate
            .get("held_out_validation")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"status":"not_recorded"})),
        consumers: Vec::new(),
        plots: calibration_plots(id, &live_estimate, previous.as_ref(), now_ms),
        notes,
        live_estimate,
        trust_state,
        confidence,
    }
}

fn attach_estimator_semantics(view: &mut CalibrationEstimatorView) {
    let trusted = view.trust_state == "trusted";
    let all_observable = !view.degrees_of_freedom.is_empty()
        && view.degrees_of_freedom.iter().all(|dof| dof.observable);
    let gate = |consumer: &str, allowed: bool, reason: &str| CalibrationConsumerGate {
        consumer: consumer.to_string(),
        allowed,
        reason: reason.to_string(),
    };
    match view.id.as_str() {
        "kinect_geometry" => {
            view.consumers.push(gate(
                "depth association",
                trusted && all_observable,
                if trusted && all_observable {
                    "trusted full transform"
                } else {
                    "blocked: transform is not fully observable and trusted"
                },
            ));
            view.consumers.push(gate(
                "mapping",
                trusted && all_observable,
                if trusted && all_observable {
                    "trusted full transform"
                } else {
                    "blocked: calibrated geometry trust gate"
                },
            ));
            view.notes.push(
                "Lidar is optional corroboration; its absence does not independently block trust."
                    .into(),
            );
        }
        "imu" => {
            let roll_pitch = view
                .degrees_of_freedom
                .iter()
                .filter(|dof| matches!(dof.name.as_str(), "roll" | "pitch"))
                .all(|dof| dof.observable);
            let yaw = view
                .degrees_of_freedom
                .iter()
                .find(|dof| dof.name == "yaw")
                .is_some_and(|dof| dof.observable);
            view.consumers.push(gate(
                "roll/pitch correction",
                trusted && roll_pitch,
                if trusted && roll_pitch {
                    "trusted observable axes"
                } else {
                    "blocked: roll/pitch trust incomplete"
                },
            ));
            view.consumers.push(gate(
                "absolute yaw",
                trusted && yaw,
                if trusted && yaw {
                    "trusted yaw axis"
                } else {
                    "blocked: yaw is unobservable or untrusted"
                },
            ));
        }
        id if id.starts_with("timing:") => view.consumers.push(gate(
            "cross-stream association",
            trusted,
            if trusted {
                "trusted clock and latency estimate"
            } else {
                "blocked: timing estimate untrusted, stale, or unobservable"
            },
        )),
        "locomotion" => {
            view.consumers.push(gate(
                "navigation correction",
                trusted,
                if trusted {
                    "trusted learned scale"
                } else {
                    "blocked: locomotion estimate is not trusted"
                },
            ));
            view.consumers.push(gate(
                "brainstem motor/safety authority",
                false,
                "never allowed: learned calibration is advisory only",
            ));
        }
        _ => {}
    }
}

fn calibration_dofs(id: &str, value: &serde_json::Value) -> Vec<CalibrationDofView> {
    let names: &[&str] = if id == "kinect_geometry" {
        &["x", "y", "z", "roll", "pitch", "yaw"]
    } else if id == "imu" {
        &["roll", "pitch", "yaw"]
    } else {
        &[]
    };
    let observable = value
        .get("observable_dofs")
        .and_then(serde_json::Value::as_array);
    let covariance = value
        .get("covariance")
        .and_then(serde_json::Value::as_array);
    let counts = value
        .get("dof_evidence_counts")
        .and_then(serde_json::Value::as_array);
    names
        .iter()
        .enumerate()
        .map(|(index, name)| CalibrationDofView {
            name: (*name).to_string(),
            observable: observable
                .and_then(|values| values.get(index))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or_else(|| match *name {
                    "roll" | "pitch" => value
                        .get("roll_pitch_observable")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                    "yaw" => value
                        .get("yaw_axis_observable")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                    _ => false,
                }),
            covariance: covariance
                .and_then(|values| values.get(index))
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32),
            evidence_count: counts
                .and_then(|values| values.get(index))
                .and_then(serde_json::Value::as_u64),
        })
        .collect()
}

fn evidence_summary(value: &serde_json::Value) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for key in [
        "evidence_count",
        "correlated_event_count",
        "rejected_count",
        "straight_evidence_count",
        "rotation_evidence_count",
        "rejected_straight_count",
        "rejected_rotation_count",
        "total_samples",
        "stationary_samples",
        "rotation_evidence_samples",
    ] {
        if let Some(found) = value.get(key) {
            result.insert(key.to_string(), found.clone());
        }
    }
    serde_json::Value::Object(result)
}

fn configured_thresholds(value: &serde_json::Value) -> serde_json::Value {
    let Some(object) = value.as_object() else {
        return serde_json::Value::Null;
    };
    serde_json::Value::Object(
        object
            .iter()
            .filter(|(key, _)| {
                key.contains("minimum")
                    || key.contains("maximum")
                    || key.contains("threshold")
                    || key.contains("stale")
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn calibration_plots(
    id: &str,
    current: &serde_json::Value,
    previous: Option<&serde_json::Value>,
    now_ms: u64,
) -> Vec<CalibrationPlotSeries> {
    let specs: &[(&str, &str, Option<&str>)] = match id {
        "kinect_geometry" => &[
            ("mount x", "/transform/translation_m/0", Some("m")),
            ("mount y", "/transform/translation_m/1", Some("m")),
            ("mount z", "/transform/translation_m/2", Some("m")),
            ("mount roll", "/transform/rotation_rpy_rad/0", Some("rad")),
            ("mount pitch", "/transform/rotation_rpy_rad/1", Some("rad")),
            ("mount yaw", "/transform/rotation_rpy_rad/2", Some("rad")),
            ("confidence", "/confidence", None),
        ],
        "imu" => &[
            ("gyro bias x", "/gyro_bias_rad_s/0", Some("rad/s")),
            ("gyro bias y", "/gyro_bias_rad_s/1", Some("rad/s")),
            ("gyro bias z", "/gyro_bias_rad_s/2", Some("rad/s")),
            ("gyro noise x", "/gyro_variance/0", None),
            ("yaw rate scale", "/yaw_rate_scale", None),
            ("confidence", "/confidence", None),
        ],
        "locomotion" => &[
            ("left wheel scale", "/left_distance_scale/value", None),
            ("right wheel scale", "/right_distance_scale/value", None),
            ("CW rotation scale", "/clockwise_rotation_scale/value", None),
            (
                "CCW rotation scale",
                "/counter_clockwise_rotation_scale/value",
                None,
            ),
            ("wheelbase", "/effective_wheelbase_m/value", Some("m")),
            ("confidence", "/confidence", None),
        ],
        _ => &[
            ("median latency", "/transport_latency/median_ms", Some("ms")),
            ("p95 latency", "/transport_latency/p95_ms", Some("ms")),
            ("jitter", "/transport_latency/jitter_ms", Some("ms")),
            (
                "latency uncertainty",
                "/transport_latency/uncertainty_ms",
                Some("ms"),
            ),
            (
                "correlated offset",
                "/correlated_offset/median_ms",
                Some("ms"),
            ),
            ("confidence", "/confidence", None),
        ],
    };
    specs
        .iter()
        .filter_map(|(metric, pointer, unit)| {
            let current_value = current
                .pointer(pointer)
                .and_then(serde_json::Value::as_f64)?;
            let mut points = Vec::new();
            if let Some(previous_value) = previous
                .and_then(|value| value.pointer(pointer))
                .and_then(serde_json::Value::as_f64)
            {
                points.push([now_ms.saturating_sub(1) as f64, previous_value]);
            }
            points.push([now_ms as f64, current_value]);
            Some(CalibrationPlotSeries {
                metric: (*metric).into(),
                unit: unit.map(str::to_string),
                points,
            })
        })
        .collect()
}

fn epoch_string(value: &serde_json::Value) -> Option<String> {
    value
        .get("epoch")
        .map(|epoch| epoch.get("id").unwrap_or(epoch))
        .and_then(|epoch| {
            epoch
                .as_u64()
                .map(|v| v.to_string())
                .or_else(|| epoch.as_str().map(str::to_string))
        })
}
fn value_string(value: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_str))
        .map(str::to_string)
}
fn value_u64(value: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_u64))
}
fn value_f64(value: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_f64))
}
fn value_f32(value: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    value_f64(value, keys).map(|v| v as f32)
}
fn string_array(value: &serde_json::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

async fn get_observatory_calibration(
    State(state): State<LiveViewState>,
    Query(query): Query<CalibrationConsoleQuery>,
) -> Result<Json<CalibrationConsoleResponse>, ObservatoryHttpError> {
    let selection = match (query.snapshot_id.as_deref(), query.at_or_before_ms) {
        (Some(id), None) => state.observatory_now_snapshot(id),
        (None, Some(t_ms)) => state.observatory_now_at_or_before(t_ms),
        _ => {
            return Err(ObservatoryHttpError::bad_request(
                "provide exactly one of snapshot_id or at_or_before_ms",
            ))
        }
    }
    .ok_or_else(|| {
        ObservatoryHttpError::unavailable("requested calibration snapshot is not retained")
    })?;
    let history = state
        .observatory()
        .query_async(BrainEventQuery {
            event_type: Some(BrainEventType::CalibrationTransition),
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..Default::default()
        })
        .await
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let events: Vec<BrainEvent> = history
        .records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => envelope.event,
            BrainEventStreamRecord::Gap { gap } => gap.event,
        })
        .collect();
    Ok(Json(build_calibration_console(&selection, &events)))
}
