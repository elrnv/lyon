use gfx;
use gfx_device_gl;
//use gfx::Factory;

use tessellation;
use tessellation::geometry_builder::VertexConstructor;
use core::math::*;
use buffer::*;

use std;
use std::mem;

pub type OpaquePso = Pso<opaque_fill_pipeline::Meta>;
pub type TransparentPso = Pso<transparent_fill_pipeline::Meta>;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;

pub type CmdEncoder = gfx::Encoder<gfx_device_gl::Resources, gfx_device_gl::CommandBuffer>;
pub type BufferObject<T> = gfx::handle::Buffer<gfx_device_gl::Resources, T>;
pub type Vbo<T> = gfx::handle::Buffer<gfx_device_gl::Resources, T>;
pub type Ibo = gfx::IndexBuffer<gfx_device_gl::Resources>;
pub type Pso<T> = gfx::PipelineState<gfx_device_gl::Resources, T>;
pub type IndexSlice = gfx::Slice<gfx_device_gl::Resources>;
pub type ColorTarget = gfx::handle::RenderTargetView<gfx_device_gl::Resources,
                                                     (gfx::format::R8_G8_B8_A8,
                                                      gfx::format::Unorm)>;
pub type DepthTarget = gfx::handle::DepthStencilView<gfx_device_gl::Resources,
                                                     (gfx::format::D24_S8, gfx::format::Unorm)>;
pub type GlDevice = gfx_device_gl::Device;
pub type GlFactory = gfx_device_gl::Factory;


gfx_defines!{
    constant Globals {
        resolution: [f32; 2] = "u_resolution",
        scroll_offset: [f32; 2] = "u_scroll_offset",
        zoom: f32 = "u_zoom",
    }

    constant GpuTransform {
        transform: [[f32; 4]; 4] = "transform",
    }

    // Per-vertex data.
    vertex GpuFillVertex {
        position: [f32; 2] = "a_position",
        normal: [f32; 2] = "a_normal",
        prim_id: i32 = "a_prim_id", // An id pointing to the PrimData struct above.
    }

    // Per fill primitive data.
    constant GpuFillPrimitive {
        color: [f32; 4] = "color",
        z_index: f32 = "z_index",
        local_transform: i32 = "local_transform",
        view_transform: i32 = "view_transform",
        width: f32 = "width",
    }

    // Per-vertex data.
    vertex GpuStrokeVertex {
        position: [f32; 2] = "a_position",
        normal: [f32; 2] = "a_normal",
        advancement: f32 = "a_advancement",
        prim_id: i32 = "a_prim_id", // An id pointing to the PrimData struct above.
    }

    // Per stroke primitive data.
    constant GpuStrokePrimitive {
        color: [f32; 4] = "color",
        z_index: f32 = "z_index",
        local_transform: i32 = "local_transform",
        view_transform: i32 = "view_transform",
        width: f32 = "width",
    }

    pipeline opaque_fill_pipeline {
        vbo: gfx::VertexBuffer<GpuFillVertex> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "out_color",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        constants: gfx::ConstantBuffer<Globals> = "Globals",
        transforms: gfx::ConstantBuffer<GpuTransform> = "u_transforms",
        primitives: gfx::ConstantBuffer<GpuFillPrimitive> = "u_primitives",
    }

    pipeline transparent_fill_pipeline {
        vbo: gfx::VertexBuffer<GpuFillVertex> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "out_color",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_TEST,
        constants: gfx::ConstantBuffer<Globals> = "Globals",
        transforms: gfx::ConstantBuffer<GpuTransform> = "u_transforms",
        primitives: gfx::ConstantBuffer<GpuFillPrimitive> = "u_primitives",
    }

    pipeline opaque_stroke_pipeline {
        vbo: gfx::VertexBuffer<GpuStrokeVertex> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "out_color",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        constants: gfx::ConstantBuffer<Globals> = "Globals",
        transforms: gfx::ConstantBuffer<GpuTransform> = "u_transforms",
        primitives: gfx::ConstantBuffer<GpuStrokePrimitive> = "u_primitives",
    }

    pipeline transparent_stroke_pipeline {
        vbo: gfx::VertexBuffer<GpuStrokeVertex> = (),
        out_color: gfx::RenderTarget<ColorFormat> = "out_color",
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_TEST,
        constants: gfx::ConstantBuffer<Globals> = "Globals",
        transforms: gfx::ConstantBuffer<GpuTransform> = "u_transforms",
        primitives: gfx::ConstantBuffer<GpuStrokePrimitive> = "u_primitives",
    }
}

impl GpuFillPrimitive {
    pub fn new(
        color: [f32; 4],
        z_index: f32,
        local_transform: TransformId,
        view_transform: TransformId,
    ) -> GpuFillPrimitive {
        GpuFillPrimitive {
            color: color,
            z_index: z_index,
            local_transform: local_transform.to_i32(),
            view_transform: view_transform.to_i32(),
            width: 0.0,
        }
    }
}

impl std::default::Default for GpuFillPrimitive {
    fn default() -> Self {
        GpuFillPrimitive::new([1.0, 1.0, 1.0, 1.0], 0.0, TransformId::new(0), TransformId::new(0))
    }
}

impl GpuStrokePrimitive {
    pub fn new(
        color: [f32; 4],
        z_index: f32,
        local_transform: TransformId,
        view_transform: TransformId,
    ) -> GpuStrokePrimitive {
        GpuStrokePrimitive {
            color: color,
            z_index: z_index,
            local_transform: local_transform.to_i32(),
            view_transform: view_transform.to_i32(),
            width: 1.0,
        }
    }
}

impl std::default::Default for GpuStrokePrimitive {
    fn default() -> Self {
        GpuStrokePrimitive::new([1.0, 1.0, 1.0, 1.0], 0.0, TransformId::new(0), TransformId::new(0))
    }
}


pub type TransformId = Id<GpuTransform>;

impl std::default::Default for GpuTransform {
    fn default() -> Self {
        GpuTransform {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl GpuTransform {
    pub fn new(mat: Transform3D) -> Self { GpuTransform { transform: mat.to_row_arrays() } }

    pub fn as_mat4(&self) -> &Transform3D { unsafe { mem::transmute(self) } }

    pub fn as_mut_mat4(&mut self) -> &mut Transform3D { unsafe { mem::transmute(self) } }
}

pub type FillPrimitiveId = Id<GpuFillPrimitive>;
pub type StrokePrimitiveId = Id<GpuStrokePrimitive>;

/// This vertex constructor forwards the positions and normals provided by the
/// tessellators and add a shape id.
pub struct WithId<T>(pub Id<T>);

// The fill tessellator does not implement normals yet, so this implementation
// just sets it to [0, 0], for now.
impl VertexConstructor<tessellation::FillVertex, GpuFillVertex> for WithId<GpuFillPrimitive> {
    fn new_vertex(&mut self, vertex: tessellation::FillVertex) -> GpuFillVertex {
        assert!(!vertex.position.x.is_nan());
        assert!(!vertex.position.y.is_nan());
        assert!(!vertex.normal.x.is_nan());
        assert!(!vertex.normal.y.is_nan());
        GpuFillVertex {
            position: vertex.position.to_array(),
            normal: vertex.normal.to_array(),
            prim_id: self.0.to_i32(),
        }
    }
}

impl VertexConstructor<tessellation::StrokeVertex, GpuStrokeVertex> for WithId<GpuStrokePrimitive> {
    fn new_vertex(&mut self, vertex: tessellation::StrokeVertex) -> GpuStrokeVertex {
        assert!(!vertex.position.x.is_nan());
        assert!(!vertex.position.y.is_nan());
        assert!(!vertex.normal.x.is_nan());
        assert!(!vertex.normal.y.is_nan());
        assert!(!vertex.advancement.is_nan());
        GpuStrokeVertex {
            position: vertex.position.to_array(),
            normal: vertex.normal.to_array(),
            advancement: vertex.advancement,
            prim_id: self.0.to_i32(),
        }
    }
}

pub struct RenderTarget {
    pub color: ColorTarget,
    pub depth: DepthTarget,
}

pub struct GpuGeometry<T> {
    pub vbo: Vbo<T>,
    pub ibo: IndexSlice,
}
