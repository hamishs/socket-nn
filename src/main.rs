use std::sync::Arc;

use candle_core::{DType, Device, Error, Shape, Tensor};

mod io;
mod server;
use crate::server::run_server;

type SharedWeights = Arc<Tensor>;

fn get_weights() -> SharedWeights {
    let tensor = Tensor::ones(Shape::from(&[2, 2]), DType::F64, &Device::Cpu).unwrap();
    Arc::new(tensor)
}

fn net_forward(weights: &Tensor, input_data: Tensor) -> Result<Tensor, Error> {
    weights + &input_data
}

#[tokio::main]
async fn main() {
    run_server("127.0.0.1:8080", get_weights(), net_forward)
        .await
        .unwrap();
}
