use clap::Parser;
use num::Complex;
use realfft::RealFftPlanner;
use std::path::PathBuf;

/// Takes the FFT of an audio file and a shifted version of it,
/// and splices its amplitude and the phase of its shifted version together, then runs the inverse FFT on it.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// input file (must be pcm_f32le WAV)
    input: PathBuf,

    /// output file
    output: PathBuf,

    /// size of FFT
    #[arg(short = 's', long, default_value_t = 512)]
    size: usize,

    /// how much to step (works as rudimentary speedup/slowdown if not equal to -s)
    #[arg(short = 'p', long, default_value_t = 256)]
    step: usize,

    /// samples to shift the shifted version by
    #[arg(short = 'k', long, default_value_t = 512)]
    skip: usize,
}

fn main() {
    let args = Args::parse();
    let mut plan = RealFftPlanner::<f32>::new();

    let mut input = hound::WavReader::open(args.input).unwrap();
    let spec = input.spec();
    assert!(
        spec.sample_format == hound::SampleFormat::Float,
        "input must be floating-point WAV (unfortunately)"
    );

    let input: Vec<f32> = input.samples().map(|n| n.unwrap()).collect();
    let mut input_shifted: Vec<f32> = input.clone();
    input_shifted.rotate_left(args.skip);

    let windows = input.windows(args.size).step_by(args.step);
    let windows_shifted = input_shifted.windows(args.size).step_by(args.step);
    let mut writer = hound::WavWriter::create(args.output, spec).unwrap();

    for (w, ws) in windows.zip(windows_shifted) {
        let (mut input, mut input_shift) = (w.to_owned(), ws.to_owned());
        let out = phase_switcheroo(&mut input, &mut input_shift, &mut plan, args.size);

        for s in out {
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn phase_switcheroo(
    first: &mut [f32],
    second: &mut [f32],
    plan: &mut RealFftPlanner<f32>,
    size: usize,
) -> Vec<f32> {
    let r2c = plan.plan_fft_forward(size);
    let c2r = plan.plan_fft_inverse(size);

    let (mut out_first, mut out_second, mut out_combined) = (
        r2c.make_output_vec(),
        r2c.make_output_vec(),
        c2r.make_output_vec()
    );

    let _ = r2c.process(first, &mut out_first);
    let _ = r2c.process(second, &mut out_second);

    let unsize = 1. / size as f32;
    let mut combined = std::iter::zip(out_first, out_second)
        .map(|(a, p)| p.scale(unsize * (a.norm_sqr() / p.norm_sqr()).sqrt()))
        .collect();

    let _ = c2r.process(&mut combined, &mut out_combined);
    out_combined
}
