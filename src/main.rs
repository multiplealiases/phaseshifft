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
    assert!(spec.sample_format == hound::SampleFormat::Float, "input must be floating-point WAV (unfortunately)");

    let input: Vec<f32> = input.samples().map(|n| n.unwrap()).collect();
    let input_shifted: Vec<f32> = input.iter().cloned().cycle().skip(args.skip).take(input.len()).collect();

    let windows = input.windows(args.size).step_by(args.step);
    let windows_shifted = input_shifted.windows(args.size).step_by(args.step);
    let mut writer = hound::WavWriter::create(args.output, spec).unwrap();

    for (w, ws) in windows.zip(windows_shifted) {
        let r2c = plan.plan_fft_forward(args.size);
        let (mut input, mut input_shift) = (w.to_owned(), ws.to_owned());
        let (mut out, mut out_shift) = (r2c.make_output_vec(), r2c.make_output_vec());

        let _ = r2c.process(&mut input, &mut out);
        let _ = r2c.process(&mut input_shift, &mut out_shift);

        let amplitude = out.into_iter().map(|c| c.im);
        let phase = out_shift.into_iter().map(|c| c.re);
        let mut combined: Vec<Complex<f32>> = Vec::new();

        for (re, im) in phase.zip(amplitude) {
            let c = Complex{ re, im };
            combined.push(c);
        }

        let mut combined = fft_normalize(combined);

        let c2r = plan.plan_fft_inverse(args.size);
        let mut out = c2r.make_output_vec();
        let _ = c2r.process(&mut combined, &mut out);

        for s in out {
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn fft_normalize(window: Vec<Complex<f32>>) -> Vec<Complex<f32>> {
    let size = (window.len() as f32 + 1.) * 2.;
    window.into_iter().map(|c| c.unscale(size)).collect()
}
