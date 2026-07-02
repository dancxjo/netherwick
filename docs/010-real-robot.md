# 010 Real Robot

The real robot target starts with Create 1 body support in safe, read-oriented form and grows toward slow controlled action after simulator behavior stabilizes.

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

## Startup Events and Mouth

The robot runner annotates the first real-robot `Now` with `robot.initialization` metadata: mode, body source, battery, requested/active sensors, ledger path, tick rate, dashboard, and capture destination. `EventExtractor` turns that first-tick annotation into a `RobotInitialized` event. The runtime then runs the replaceable `event_robot_initialized` behavior node, which can emit a bring-up `Song`, `Chirp`, and spoken status sequence.

The robot process owns rendering. It creates a queued Piper/CPAL mouth from:

```bash
just setup-ort
just setup-tts
# or, as part of full system setup:
just setup
```

The default voice is downloaded to the Tongues Piper model directory and autoloaded at startup. To override it, set:

```bash
NETHERWICK_TTS_PIPER_VOICE=/path/to/en_US-ryan-medium.onnx
NETHERWICK_TTS_PIPER_CONFIG=/path/to/en_US-ryan-medium.onnx.json
NETHERWICK_TTS_OUTPUT_DEVICE="USB Audio Device"
ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

Command-backed ASR uses the robot microphone and local Whisper:

```bash
just setup-whisper
MIC_DEVICE=default
NETHERWICK_WHISPER_MODEL=/path/to/ggml-base.en.bin
NETHERWICK_ASR_COMMAND=target/debug/netherwick whisper-transcribe
```

When configured, bring-up lines are enqueued immediately and played sequentially on a background thread using Tongues Piper streaming plus CPAL output. Later mouth actions emitted by event scripts are appended to the same queue. If the Piper voice or output device is unavailable, the robot should report the mouth as disabled and continue the robot run rather than blocking body/sensor startup.

Mouth actions do not command motors. `Speak`, `Chirp`, and `Song` are rendered through the mouth gate; motion primitives remain separate and still pass the real-robot mode and safety gates.
