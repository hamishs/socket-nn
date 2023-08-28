//! Example of a simple server that runs a neural network.
//!
//! Run with `rust-script`.
//!
//! ```cargo
//! [dependencies]
//! candle-core = { version = "0.1.2" }
//! tokio = { version = "1", features = ["full"] }
//! socket-nn = { path = ".." }
//! ```
use std::sync::Arc;

use candle_core::{DType, Device, Error, Shape, Tensor};

//extern crate socket_nn;
use socket_nn::server::run_server;

fn get_weights() -> Arc<Tensor> {
    let tensor = Tensor::ones(Shape::from(&[2, 2]), DType::F64, &Device::Cpu).unwrap();
    Arc::new(tensor)
}

fn net_forward(weights: &Tensor, input_data: Tensor) -> Result<Tensor, Error> {
    weights + &input_data
}

#[tokio::main]
async fn main() {
    println!("Running server on localhost 8080...");
    run_server("127.0.0.1:8080", get_weights(), net_forward)
        .await
        .unwrap();
}
