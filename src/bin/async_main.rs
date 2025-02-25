#![no_std]
#![no_main]

use burn::backend::NdArray;
use burn::tensor::Tensor;
use defmt::*;
use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use esp_hal::clock::CpuClock;
use {defmt_rtt as _, esp_backtrace as _};
extern crate alloc;
type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

#[esp_hal_embassy::main]
async fn main(spawner: Spawner) {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(8 * 1024);

    let timer0 = esp_hal::timer::systimer::SystemTimer::new(peripherals.SYSTIMER);
    esp_hal_embassy::init(timer0.alarm0);

    info!("Embassy initialized!");
    let _ = spawner;
    let mut img_array = [[[0.0; WIDTH]; HEIGHT]; 3];

    // Iterate over the pixels and populate the array
    for y in 0..224usize {
        for x in 0..224usize {
            let rgb = [0, 0, 0];
            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }
    let device = BackendDevice::default();

    let image_input =
        Tensor::<Backend, 3>::from_data(img_array, &device).reshape([1, 3, HEIGHT, WIDTH]);

    loop {
        info!("Hello world!");
        Timer::after(Duration::from_secs(1)).await;
    }
}
