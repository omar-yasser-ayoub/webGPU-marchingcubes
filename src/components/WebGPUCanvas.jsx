import React, { useEffect, useRef } from 'react';
import { mat3, mat4, vec3, vec4 } from 'wgpu-matrix';

const WebGPUCanvas = () => {
    const canvasRef = useRef(null);

    useEffect(() => {

        const canvas = canvasRef.current;

        const camera = {
            Eye: [1, 1, 1],
            Look: [0, 0, 0],
            At: [0, 1, 0],
        }
        //create an isometric projection matrix
        const projectionMatrix = mat4.ortho(-1, 1, -1, 1, 0.1, 1000);

        //create a view matrix
        const viewMatrix = mat4.lookAt(camera.Eye, camera.Look, camera.At);
        
        //skip the model matrix for now

        async function init() {
            // Check for WebGPU support
            if (!navigator.gpu) {
                console.error('WebGPU is not supported in your browser');
                return;
            }

            // Get the WebGPU context
            const context = canvas.getContext('webgpu');
            if (!context) {
                console.error('WebGPU context could not be created');
                return;
            }

            // Request an adapter and device
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error('Failed to get GPU adapter');
                return;
            }

            const device = await adapter.requestDevice();

            // Configure the canvas context
            const format = navigator.gpu.getPreferredCanvasFormat();
            


            const vertices = new Float32Array([
                // X, Y, Z coordinates
                -0.5, -0.5, -0.5,   // Vertex 0
                 0.5, -0.5, -0.5,   // Vertex 1
                 0.5,  0.5, -0.5,   // Vertex 2
                -0.5,  0.5, -0.5,   // Vertex 3
                -0.5, -0.5,  0.5,   // Vertex 4
                 0.5, -0.5,  0.5,   // Vertex 5
                 0.5,  0.5,  0.5,   // Vertex 6
                -0.5,  0.5,  0.5    // Vertex 7
            ]);
        
            const indices = new Uint32Array([
                // Front face
                0, 1, 2,  2, 3, 0,
                // Back face
                4, 5, 6,  6, 7, 4,
                // Left face
                0, 4, 7,  7, 3, 0,
                // Right face
                1, 5, 6,  6, 2, 1,
                // Top face
                3, 7, 6,  6, 2, 3,
                // Bottom face
                0, 1, 5,  5, 4, 0
            ]);


            //create buffers for the vertices and indices and copy the data to the GPU
            const vertexBuffer = device.createBuffer({
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            const indexBuffer = device.createBuffer({
                size: indices.byteLength,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
            });
            const uniformBuffer = device.createBuffer({
                size: 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            const matrixData = new Float32Array(32);
            matrixData.set(projectionMatrix, 0);
            matrixData.set(viewMatrix, 16);

            device.queue.writeBuffer(uniformBuffer, 0, matrixData.buffer, matrixData.byteOffset, matrixData.byteLength);
            device.queue.writeBuffer(vertexBuffer, 0, vertices.buffer, vertices.byteOffset, vertices.byteLength);
            device.queue.writeBuffer(indexBuffer, 0, indices.buffer, indices.byteOffset, indices.byteLength);

            const bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.VERTEX,
                        buffer: { type: 'uniform' }
                    }
                ]
            });

            const bindGroup = device.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: uniformBuffer
                        }
                    }
                ]
            });

            const depthTexture = device.createTexture({
                size: [canvas.width, canvas.height, 1],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
              });
              
            const depthView = depthTexture.createView();

            // Define shader code
            const shaderCode = `
                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) color: vec4<f32>,
                }

                struct Uniforms {
                    projectionMatrix: mat4x4<f32>,
                    viewMatrix: mat4x4<f32>,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vertexMain(@location(0) position: vec3<f32>) -> VertexOutput {
                    var output: VertexOutput;
                    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * vec4(position, 1.0);
                    output.color = vec4(position * 0.5 + 0.5, 1.0); // Generate color based on position
                    return output;
                }

                @fragment
                fn fragmentMain(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                    return color;
                }
            `;

            // Create a shader module
            const shaderModule = device.createShaderModule({
                code: shaderCode
            });
            // Create a render pipeline
            const pipeline = device.createRenderPipeline({
                layout: device.createPipelineLayout({ bindGroupLayouts: [
                    bindGroupLayout
                ] }),
                vertex: {
                    module: shaderModule,
                    entryPoint: 'vertexMain',
                    buffers: [
                        {
                            arrayStride: 4 * 3, // 3 floats (x, y, z) * 4 bytes each
                            attributes: [
                                {
                                    shaderLocation: 0,
                                    offset: 0,
                                    format: 'float32x3' // 3-component vector (x, y, z)
                                }
                            ]
                        }
                    ]
                },
                fragment: {
                    module: shaderModule,
                    entryPoint: 'fragmentMain',
                    targets: [{
                        format: format
                    }]
                },
                primitive: {
                    topology: 'triangle-list',
                    cullMode: 'none',
                  },
                depthStencil: {
                    format: 'depth24plus',
                    depthWriteEnabled: true,
                    depthCompare: 'less',
                },
            });

            // Render a frame
            function renderFrame() {
                context.configure({
                    device,
                    format,
                });
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        {
                          view: context.getCurrentTexture().createView(),
                          loadOp: 'clear',
                          storeOp: 'store',
                          clearValue: { r: 0, g: 0, b: 0, a: 1 },
                        },
                    ],
                    depthStencilAttachment: {
                    view: depthView,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                    depthClearValue: 1.0,
                    },
                });

                pass.setIndexBuffer(indexBuffer, 'uint32');
                pass.setVertexBuffer(0, vertexBuffer);
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.drawIndexed(36); // Draw a triangle
                pass.end();

                device.queue.submit([encoder.finish()]);
                requestAnimationFrame(renderFrame); // Continuously render
            }

            renderFrame();
        }

        init().catch((error) => {
            console.error('Error initializing WebGPU:', error);
        });
    }, []);

    return <canvas ref={canvasRef} width={800} height={600} />;
};

export default WebGPUCanvas;