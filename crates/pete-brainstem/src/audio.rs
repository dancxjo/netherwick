use crate::commands::{SongTone, MAX_SONG_TONES};

pub(crate) const AUTOMATIC_CUE_SLOT: u8 = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum AuditoryCue {
    None = 0,
    Armed = 1,
    EStop = 2,
    Cliff = 3,
    WheelDrop = 4,
    Tilt = 5,
    Impact = 6,
    HeartbeatLost = 7,
    AuthorityLost = 8,
    AuthorityReplaced = 9,
    BumpContact = 10,
    CreateError = 11,
    RuntimeError = 12,
    ServiceFailure = 13,
    LowBattery = 14,
    SafetyClear = 15,
    Recovery = 16,
    AuthorityAcquired = 17,
    DockContact = 18,
    ImuFault = 19,
    ServiceComplete = 20,
    DockSeen = 21,
    MotionInconsistency = 22,
}

impl AuditoryCue {
    pub(crate) const fn code(self) -> u8 {
        self as u8
    }

    pub(crate) const fn priority(self) -> u8 {
        match self {
            Self::EStop => 8,
            Self::Cliff | Self::WheelDrop | Self::Tilt | Self::Impact => 7,
            Self::HeartbeatLost
            | Self::AuthorityLost
            | Self::AuthorityReplaced
            | Self::CreateError => 6,
            Self::BumpContact => 5,
            Self::RuntimeError
            | Self::ServiceFailure
            | Self::ImuFault
            | Self::MotionInconsistency => 4,
            Self::LowBattery => 3,
            Self::SafetyClear | Self::Recovery => 2,
            Self::Armed
            | Self::AuthorityAcquired
            | Self::DockContact
            | Self::ServiceComplete
            | Self::DockSeen => 1,
            Self::None => 0,
        }
    }

    pub(crate) const fn urgent(self) -> bool {
        self.priority() >= 3
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Armed => "armed",
            Self::EStop => "estop",
            Self::Cliff => "cliff",
            Self::WheelDrop => "wheel_drop",
            Self::Tilt => "tilt",
            Self::Impact => "impact",
            Self::HeartbeatLost => "heartbeat_lost",
            Self::AuthorityLost => "authority_lost",
            Self::AuthorityReplaced => "authority_replaced",
            Self::BumpContact => "bump_contact",
            Self::CreateError => "create_error",
            Self::RuntimeError => "runtime_error",
            Self::ServiceFailure => "service_failure",
            Self::LowBattery => "low_battery",
            Self::SafetyClear => "safety_clear",
            Self::Recovery => "recovery",
            Self::AuthorityAcquired => "authority_acquired",
            Self::DockContact => "dock_contact",
            Self::ImuFault => "imu_fault",
            Self::ServiceComplete => "service_complete",
            Self::DockSeen => "dock_seen",
            Self::MotionInconsistency => "motion_inconsistency",
        }
    }
}

pub(crate) fn cue_name(code: u8) -> &'static str {
    cue_from_code(code).name()
}

fn cue_from_code(code: u8) -> AuditoryCue {
    match code {
        1 => AuditoryCue::Armed,
        2 => AuditoryCue::EStop,
        3 => AuditoryCue::Cliff,
        4 => AuditoryCue::WheelDrop,
        5 => AuditoryCue::Tilt,
        6 => AuditoryCue::Impact,
        7 => AuditoryCue::HeartbeatLost,
        8 => AuditoryCue::AuthorityLost,
        9 => AuditoryCue::AuthorityReplaced,
        10 => AuditoryCue::BumpContact,
        11 => AuditoryCue::CreateError,
        12 => AuditoryCue::RuntimeError,
        13 => AuditoryCue::ServiceFailure,
        14 => AuditoryCue::LowBattery,
        15 => AuditoryCue::SafetyClear,
        16 => AuditoryCue::Recovery,
        17 => AuditoryCue::AuthorityAcquired,
        18 => AuditoryCue::DockContact,
        19 => AuditoryCue::ImuFault,
        20 => AuditoryCue::ServiceComplete,
        21 => AuditoryCue::DockSeen,
        22 => AuditoryCue::MotionInconsistency,
        _ => AuditoryCue::None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CueRequestResult {
    Ready,
    Queued,
    Suppressed,
    Dropped,
}

pub(crate) struct AudioAnnunciator {
    silent: bool,
    urgent: Option<AuditoryCue>,
    informational: Option<AuditoryCue>,
    busy_until_ms: u32,
}

impl AudioAnnunciator {
    pub(crate) const fn new() -> Self {
        Self {
            silent: false,
            urgent: None,
            informational: None,
            busy_until_ms: 0,
        }
    }

    pub(crate) fn silent(&self) -> bool {
        self.silent
    }

    pub(crate) fn set_silent(&mut self, silent: bool) -> u32 {
        self.silent = silent;
        if !silent {
            return 0;
        }
        let dropped = u32::from(self.urgent.take().is_some())
            + u32::from(self.informational.take().is_some());
        dropped
    }

    pub(crate) fn request(&mut self, cue: AuditoryCue, now_ms: u32) -> CueRequestResult {
        if self.silent {
            return CueRequestResult::Suppressed;
        }
        if !time_reached(now_ms, self.busy_until_ms) {
            return self.enqueue(cue);
        }
        let replaced_information = cue.urgent() && self.informational.take().is_some();
        if self.urgent.is_none() && self.informational.is_none() {
            if cue.urgent() {
                self.urgent = Some(cue);
            } else {
                self.informational = Some(cue);
            }
            if replaced_information {
                CueRequestResult::Dropped
            } else {
                CueRequestResult::Ready
            }
        } else {
            self.enqueue(cue)
        }
    }

    fn enqueue(&mut self, cue: AuditoryCue) -> CueRequestResult {
        if cue.urgent() {
            let replaced_information = self.informational.take().is_some();
            match self.urgent {
                Some(pending) if pending.priority() > cue.priority() => CueRequestResult::Dropped,
                Some(_) => {
                    self.urgent = Some(cue);
                    CueRequestResult::Dropped
                }
                None => {
                    self.urgent = Some(cue);
                    if replaced_information {
                        CueRequestResult::Dropped
                    } else {
                        CueRequestResult::Queued
                    }
                }
            }
        } else {
            let replaced = self.informational.replace(cue).is_some();
            if replaced {
                CueRequestResult::Dropped
            } else {
                CueRequestResult::Queued
            }
        }
    }

    pub(crate) fn take_ready(&mut self, now_ms: u32) -> Option<AuditoryCue> {
        if self.silent || !time_reached(now_ms, self.busy_until_ms) {
            return None;
        }
        self.urgent.take().or_else(|| self.informational.take())
    }

    pub(crate) fn playback_available(&self, now_ms: u32) -> bool {
        !self.silent
            && self.urgent.is_none()
            && self.informational.is_none()
            && time_reached(now_ms, self.busy_until_ms)
    }

    pub(crate) fn mark_manual_played(&mut self, now_ms: u32, duration_ms: u32) {
        self.busy_until_ms = now_ms.wrapping_add(duration_ms.max(1));
    }

    pub(crate) fn mark_played(&mut self, cue: AuditoryCue, now_ms: u32) {
        self.busy_until_ms = now_ms.wrapping_add(cue_duration_ms(cue));
    }

    #[cfg(test)]
    pub(crate) fn pending_counts(&self) -> (usize, usize) {
        (
            usize::from(self.urgent.is_some()),
            usize::from(self.informational.is_some()),
        )
    }
}

pub(crate) fn cue_tones(cue: AuditoryCue) -> ([SongTone; MAX_SONG_TONES], u8) {
    // C4..B4 are Pete's do, re, mi, fa, sol, la, si pitch classes. The
    // sequences are stable operational codes; they are not presented as
    // grammatical Solresol vocabulary.
    const DO: u8 = 60;
    const RE: u8 = 62;
    const MI: u8 = 64;
    const FA: u8 = 65;
    const SOL: u8 = 67;
    const LA: u8 = 69;
    const SI: u8 = 71;
    let notes: &[u8] = match cue {
        AuditoryCue::Armed => &[FA, SOL, SI],
        AuditoryCue::EStop => &[DO, DO, DO, DO],
        AuditoryCue::Cliff => &[SI, DO, SI, DO],
        AuditoryCue::WheelDrop => &[DO, FA, DO, FA],
        AuditoryCue::Tilt => &[LA, FA, RE],
        AuditoryCue::Impact => &[DO, SOL, DO],
        AuditoryCue::HeartbeatLost => &[SOL, MI, DO],
        AuditoryCue::AuthorityLost => &[SOL, RE, DO],
        AuditoryCue::AuthorityReplaced => &[SOL, FA, RE],
        AuditoryCue::BumpContact => &[DO, RE, DO],
        AuditoryCue::CreateError => &[MI, RE, DO, DO],
        AuditoryCue::RuntimeError => &[MI, RE, DO],
        AuditoryCue::ServiceFailure => &[FA, RE, DO],
        AuditoryCue::LowBattery => &[LA, MI, DO],
        AuditoryCue::SafetyClear => &[DO, MI, SOL],
        AuditoryCue::Recovery => &[RE, FA, LA],
        AuditoryCue::AuthorityAcquired => &[DO, SOL, MI],
        AuditoryCue::DockContact => &[SOL, SI, RE],
        AuditoryCue::ImuFault => &[SI, FA, MI],
        AuditoryCue::ServiceComplete => &[RE, SOL, SI],
        AuditoryCue::DockSeen => &[SOL, RE, SI],
        AuditoryCue::MotionInconsistency => &[MI, SI, MI],
        AuditoryCue::None => &[],
    };
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    for (index, note) in notes.iter().enumerate() {
        tones[index] = SongTone {
            note: *note,
            duration_64ths: if index + 1 == notes.len() { 10 } else { 6 },
        };
    }
    (tones, notes.len() as u8)
}

fn cue_duration_ms(cue: AuditoryCue) -> u32 {
    let (tones, count) = cue_tones(cue);
    tone_duration_ms(&tones, count)
}

pub(crate) fn tone_duration_ms(tones: &[SongTone; MAX_SONG_TONES], count: u8) -> u32 {
    tones[..count.min(MAX_SONG_TONES as u8) as usize]
        .iter()
        .map(|tone| u32::from(tone.duration_64ths) * 1_000 / 64)
        .sum::<u32>()
        .max(1)
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hazard_motifs_are_unique_and_use_solresol_pitch_classes() {
        let hazards = [
            AuditoryCue::EStop,
            AuditoryCue::Cliff,
            AuditoryCue::WheelDrop,
            AuditoryCue::Tilt,
            AuditoryCue::Impact,
            AuditoryCue::HeartbeatLost,
            AuditoryCue::BumpContact,
        ];
        for (index, cue) in hazards.iter().enumerate() {
            let (tones, count) = cue_tones(*cue);
            assert!(tones[..count as usize]
                .iter()
                .all(|tone| matches!(tone.note % 12, 0 | 2 | 4 | 5 | 7 | 9 | 11)));
            for other in &hazards[index + 1..] {
                assert!(cue_tones(*cue) != cue_tones(*other));
            }
        }
    }

    #[test]
    fn scheduler_is_bounded_and_urgent_replaces_information() {
        let mut audio = AudioAnnunciator::new();
        audio.busy_until_ms = 1_000;
        assert_eq!(
            audio.request(AuditoryCue::DockSeen, 10),
            CueRequestResult::Queued
        );
        assert_eq!(
            audio.request(AuditoryCue::AuthorityAcquired, 20),
            CueRequestResult::Dropped
        );
        assert_eq!(
            audio.request(AuditoryCue::LowBattery, 30),
            CueRequestResult::Dropped
        );
        assert_eq!(audio.pending_counts(), (1, 0));
        assert_eq!(
            audio.request(AuditoryCue::EStop, 40),
            CueRequestResult::Dropped
        );
        assert_eq!(audio.pending_counts(), (1, 0));
        assert_eq!(audio.take_ready(1_000), Some(AuditoryCue::EStop));
    }

    #[test]
    fn silent_discards_pending_and_never_replays_it() {
        let mut audio = AudioAnnunciator::new();
        audio.busy_until_ms = 1_000;
        audio.request(AuditoryCue::DockSeen, 10);
        assert_eq!(audio.set_silent(true), 1);
        assert_eq!(
            audio.request(AuditoryCue::EStop, 20),
            CueRequestResult::Suppressed
        );
        assert_eq!(audio.set_silent(false), 0);
        assert_eq!(audio.take_ready(1_000), None);
    }
}
