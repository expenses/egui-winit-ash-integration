# egui-winit-ash-integration

[![Latest version](https://img.shields.io/crates/v/egui-winit-ash-integration.svg)](https://crates.io/crates/egui-winit-ash-integration)
[![Documentation](https://docs.rs/egui-winit-ash-integration/badge.svg)](https://docs.rs/egui-winit-ash-integration)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)
[![egui version: 0.23.0](https://img.shields.io/badge/egui%20version-0.23.0-orange)](https://docs.rs/egui/0.23.0/egui/index.html)

This is the [egui](https://github.com/emilk/egui) integration crate for [egui-winit](https://github.com/emilk/egui/tree/master/crates/egui-winit) and [ash](https://github.com/MaikKlein/ash).
The default GPU allocator is [gpu_allocator](https://github.com/Traverse-Research/gpu-allocator), but you can also implement AllocatorTrait.

This crate does not support multi-viewports for multi-window since version 0.24 of egui.

[egui-ash](https://github.com/MatchaChoco010/egui-ash) is an alternative crate, which supports multi-viewports, so please take a look at this one.

# Example

```sh
cargo run --example example
```

```sh
cargo run --example user_texture
```

# Usage

```rust
fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    // (1) Call Integration::<Arc<Mutex<Allocator>>>::new() in App::new().
    let mut app = App::new(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, window_id: _ } => {
                // (2) Call integration.handle_event(&event).
                let _response = app.egui_integration.handle_event(&event);
                match event {
                    WindowEvent::Resized(_) => {
                        app.recreate_swapchain().unwrap();
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        // (3) Call integration.recreate_swapchain(...) in app.recreate_swapchain().
                        app.recreate_swapchain().unwrap();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            },
            Event::MainEventsCleared => app.window.request_redraw(),
            Event::RedrawRequested(_window_id) => {
                // (4) Call integration.begin_frame(), integration.end_frame(&mut window),
                // integration.context().tessellate(shapes), integration.paint(...)
                // in app.draw().
                app.draw().unwrap();
            },
            _ => (),
        }
    })
}
// (5) Call integration.destroy() when drop app.
```

[Full example is in examples directory](https://github.com/MatchaChoco010/egui-winit-ash-integration/tree/main/examples)

# Feature flags

`gpu-allocator-feature` - Enables the gpu-allocator crate.

The other features directly control the underlying [egui_winit features](https://docs.rs/egui-winit/latest/egui_winit/)

# License

MIT OR Apache-2.0
