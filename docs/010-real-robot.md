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
