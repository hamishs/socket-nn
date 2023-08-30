//! Example of a simple server that runs a neural network.
//!
//! Run with `cargo run --example mlp`.
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use socket_nn::server::run_server;

struct Linear {
    weight: Tensor,
    bias: Tensor,
}
impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn load_model() -> Arc<Model> {
    let device = Device::Cpu;
    let weight = Tensor::zeros((32, 32), DType::F64, &device).unwrap();
    let bias = Tensor::zeros((32,), DType::F64, &device).unwrap();
    let first = Linear { weight, bias };
    let weight = Tensor::zeros((32, 10), DType::F64, &device).unwrap();
    let bias = Tensor::zeros((10,), DType::F64, &device).unwrap();
    let second = Linear { weight, bias };
    let model = Model { first, second };
    Arc::new(model)
}

fn forward(model: &Model, input_data: Tensor) -> Result<Tensor> {
    model.forward(&input_data)
}

#[tokio::main]
async fn main() {
    println!("Running server on localhost 8080...");
    run_server("127.0.0.1:8080", load_model(), forward)
        .await
        .unwrap();
}
