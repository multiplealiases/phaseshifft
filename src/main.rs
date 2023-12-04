use clap::{Args, Parser, Subcommand};
use realfft::RealFftPlanner;
use std::path::PathBuf;

/// Takes the FFT of an audio file and a shifted version of it,
/// and splices its amplitude and the phase of its shifted version together, then runs the inverse FFT on it.
#[derive(Parser, Debug, Clone)]
#[clap(version)]
#[command(author, version, about)]
pub struct Arg {
    #[clap(subcommand)]
    mode: Mode,

    #[clap(flatten)]
    global_args: GlobalArgs,

}

#[derive(Debug, Args, Clone)]
#[command(flatten = true)]
struct GlobalArgs {
    /// size of FFT
    #[arg(short = 's', global = true, default_value_t = 512)]
    size: usize,

    /// how much to step (works as rudimentary speedup/slowdown if not equal to -s)
    #[arg(short = 'p', global = true, default_value_t = 256)]
    step: usize,

    /// samples to shift the shifted version by
    #[arg(short = 'k', global = true, default_value_t = 128)]
    skip: usize,
}

#[derive(Subcommand, Debug, Clone)]
enum Mode {
    /// transplant phase from a shifted-ahead version of the same file
    #[command(flatten = true)]
    Shift {
        /// first input file (must be pcm_f32le WAV)
        #[arg(flatten = true)]
        input: PathBuf,

        /// output file
        #[arg(flatten = true)]
        output: PathBuf
    },
    /// transplant phase data from a second file into the first
    #[command(flatten = true)]
    Transplant {
        /// first input file (must be pcm_f32le WAV)
        #[arg(flatten = true)]
        first: PathBuf,

        /// second input file (must be pcm_f32le WAV)
        #[arg(flatten = true)]
        second: PathBuf,

        /// output file
        #[arg(flatten = true)]
        output: PathBuf
    },
}

fn main() {
    let args = Arg::parse();
    let mut plan = RealFftPlanner::<f32>::new();

    match args.mode {
        Mode::Shift {
            ref input,
            ref output,
        } => {
            let (input, input_spec) = init_audio(input.to_path_buf());
            run_shift(
                args.clone(),
                input,
                output.to_path_buf(),
                input_spec,
                &mut plan,
            );
        }
        Mode::Transplant {
            ref first,
            ref second,
            ref output,
        } => {
            let (first, first_spec) = init_audio(first.to_path_buf());
            let (second, _) = init_audio(second.to_path_buf());
            run_transplant(
                args.clone(),
                first,
                second,
                output.to_path_buf(),
                first_spec,
                &mut plan,
            );
        }
    }
}

fn init_audio(path: PathBuf) -> (Vec<f32>, hound::WavSpec) {
    let mut a = hound::WavReader::open(path.clone()).unwrap();
    let spec = a.spec();
    let a: Vec<f32> = a.samples().map(|n| n.unwrap()).collect();
    (a, spec)
}

fn run_transplant(
    args: Arg,
    first: Vec<f32>,
    second: Vec<f32>,
    output: PathBuf,
    spec: hound::WavSpec,
    plan: &mut RealFftPlanner<f32>,
) {
    let first_windows = first
        .windows(args.global_args.size)
        .step_by(args.global_args.step);
    let second_windows = second
        .windows(args.global_args.size)
        .step_by(args.global_args.step);

    let mut writer = hound::WavWriter::create(output, spec).unwrap();
    for (f, s) in first_windows.zip(second_windows) {
        let (mut input, mut input_shift) = (f.to_owned(), s.to_owned());
        let out = phase_switcheroo(&mut input, &mut input_shift, plan, args.global_args.size);

        for s in out {
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn run_shift(
    args: Arg,
    input: Vec<f32>,
    output: PathBuf,
    spec: hound::WavSpec,
    plan: &mut RealFftPlanner<f32>,
) {
    let mut input_shifted: Vec<f32> = input.clone();
    input_shifted.rotate_left(args.global_args.skip);

    let windows = input
        .windows(args.global_args.size)
        .step_by(args.global_args.step);
    let windows_shifted = input_shifted
        .windows(args.global_args.size)
        .step_by(args.global_args.step);

    let mut writer = hound::WavWriter::create(output, spec).unwrap();
    for (w, ws) in windows.zip(windows_shifted) {
        let (mut input, mut input_shift) = (w.to_owned(), ws.to_owned());
        let out = phase_switcheroo(&mut input, &mut input_shift, plan, args.global_args.size);

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
    out_first
        .iter_mut()
        .zip(out_second)
        .for_each(|(a, p)| *a = p.scale(unsize * (a.norm_sqr() / p.norm_sqr()).sqrt()));

    let _ = c2r.process(&mut out_first, &mut out_combined);
    out_combined
}
