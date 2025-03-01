#![no_std]
#![no_main]

use burn::backend::NdArray;
use burn::tensor::Tensor;
use squeezenet_burn::model::{label::LABELS, normalizer::Normalizer, squeezenet1::Model};

use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_println::println;
extern crate alloc;
type Backend = NdArray<f32>;
// type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

const HEIGHT: usize = 12;
const WIDTH: usize = 12;

#[esp_hal_embassy::main]
async fn main(spawner: Spawner) {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(12 * 12 * 3 * 16);
    let timer0 = esp_hal::timer::systimer::SystemTimer::new(peripherals.SYSTIMER);
    esp_hal_embassy::init(timer0.alarm0);

    println!("Embassy initialized!");
    let _ = spawner;
    let mut img_array = [[[0.0; WIDTH]; HEIGHT]; 3];

    // Iterate over the pixels and populate the array
    for y in 0..12usize {
        for x in 0..12usize {
            let rgb = [0, 0, 0];
            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }
    // let device = BackendDevice::default();
    let device = Default::default();

    let image_input =
        Tensor::<Backend, 3>::from_data(img_array, &device).reshape([1, 3, HEIGHT, WIDTH]);

    let normalizer = Normalizer::new(&device);
    let normalized_image = normalizer.normalize(image_input);

    // Load model from embedded weights
    let model = Model::<Backend>::from_embedded();

    // Run the model
    let output = model.forward(normalized_image);

    // Get the argmax of the output
    let arg_max = output.argmax(1).into_scalar() as usize;

    // Get the label from the argmax
    let label = LABELS[arg_max];

    println!("Predicted label: {}", label);

    loop {
        println!("Hello world!");
        Timer::after(Duration::from_secs(1)).await;
    }
}
