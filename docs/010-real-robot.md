# 010 Real Robot

The real robot target starts with brainstem Cockpit support in safe, read-oriented form and grows toward slow controlled action after simulator behavior stabilizes.

Linux hardware setup is driven from the repo `Justfile`.

```bash
just setup
just hardware-env
```

For Kinect 1, the default setup path installs `libfreenect` userspace support:

```bash
just setup-kinect
```

If distro packages are missing, build from source:

```bash
just setup-kinect-from-source
```

## Startup Events and Robot Output

The robot runner annotates the first real-robot `Now` with `robot.initialization` metadata: mode, cockpit source, battery, requested/active sensors, ledger path, tick rate, dashboard, and capture destination. `EventExtractor` turns that first-tick annotation into a `RobotInitialized` event. The runtime then runs the replaceable `event_robot_initialized` behavior node, which can emit a bring-up `Song`, `Chirp`, and spoken status sequence.

The robot process owns rendering. It creates a queued Tongues/CPAL mouth from:

```bash
just setup-tts
# or, as part of full system setup:
just setup
```

The default voice is downloaded to the Tongues voice model directory and autoloaded at startup. Native Burn component and VITS models use CUDA automatically when it is available, including Burn graph fusion and kernel autotuning, and otherwise fall back to CPU. Set `PETE_TTS_COMPUTE=cpu` to force CPU or `PETE_TTS_COMPUTE=cuda` to require CUDA. The ONNX compatibility backend uses the self-contained Rust `ort` package prepared by `just setup-ort`.

To use a multi-speaker VITS model, set:

```bash
PETE_TTS_MODEL=/path/to/vits/model_file.pth
PETE_TTS_CONFIG=/path/to/vits/config.json
PETE_TTS_SPEAKERS=/path/to/vits/speaker_ids.json
PETE_TTS_SPEAKER=p225
PETE_TTS_COMPUTE=auto
PETE_TTS_OUTPUT_DEVICE=
```

To use separate acoustic and vocoder components, set:

```bash
PETE_TTS_ACOUSTIC_MODEL=/path/to/speedy-speech/model_file.pth
PETE_TTS_ACOUSTIC_CONFIG=/path/to/speedy-speech/config.json
PETE_TTS_VOCODER_MODEL=/path/to/hifigan-v2/model_file.pth
PETE_TTS_VOCODER_CONFIG=/path/to/hifigan-v2/config.json
PETE_TTS_COMPUTE=auto
PETE_TTS_OUTPUT_DEVICE=
```

Command-backed ASR uses the robot microphone and local Whisper:

```bash
just setup-whisper
MIC_DEVICE=default
PETE_WHISPER_MODEL=/path/to/ggml-tiny.en.bin
PETE_ASR_COMMAND=target/debug/pete whisper-transcribe
```

When configured, spoken bring-up lines are enqueued immediately and played sequentially on a background thread using Tongues speech streaming plus CPAL output. `Song` and `Chirp` actions are rendered through Cockpit feedback/song verbs when the backend supports them. Later spoken actions emitted by event scripts are appended to the mouth queue. If the voice model or output device is unavailable, the robot should report the mouth as disabled and continue the robot run rather than blocking cockpit/sensor startup.

Mouth and body-audio actions do not command motors. `Speak` is rendered through the mouth gate; `Chirp` and `Song` use the body-audio gate; motion primitives remain separate and still pass the real-robot mode and safety gates.
