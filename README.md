# phaseshifft
Takes the FFT of an audio file and a shifted version of it, and splices its amplitude and the phase of its shifted version together, then inverts that and produces a WAV file.

# Pre-Usage

Only accepts floating-point WAV as input, unfortunately. To convert, run either one of:

* `ffmpeg -i input.flac -c:a pcm_f32le input.wav`
* `sox input.flac -e floating-point -b 32 input.wav`

# Usage

(replace `phaseshifft` with `cargo run --release --` if you're running inside the current dir)

2x speedup:

```
$ phaseshifft input.flac output.flac --skip 0 --size 512 --step 1024
```

Slow down by half:

```
$ phaseshifft input.flac output.flac --skip 0 --size 1024 --step 512
```

Do horrible things to the phase, something something introduces noise:

```
$ phaseshifft input.flac output.flac --skip 1024 --size 512 --step 512
$ phaseshifft input.flac output.flac --skip 44100 --size 512 --step 512
```

Why not both?

```
$ phaseshifft input.flac output.flac --skip 1024 --size 512 --step 256
```
