import React, { useEffect, useRef } from 'react';
import { mat3, mat4, vec3, vec4 } from 'wgpu-matrix';
import { createNoise3D } from 'simplex-noise'
import { EdgeMasks, Edges, TriangleTable, Points } from './marchingCubesLookup';

const WebGPUCanvas = () => {
    const canvasRef = useRef(null);

    const gridSize = 64;
    const threshold = 128;

    function random(x,y,z) {
        return x*x + y*y + z*z - 0.75*0.75;
    }
    
    const voxel_grid = new Float32Array(gridSize * gridSize * gridSize);

    const positions = []

    function populateVoxelGrid() {
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    const value = random(x,y,z);
                    voxel_grid[x + y * gridSize + z * gridSize * gridSize] = value;
                }
            }
        }
    }

    function readFromGrid(x, y, z) {
        return voxel_grid[x + y * gridSize + z * gridSize * gridSize];
    }

    function getTriangulation(x, y, z) {
        let cubeIndex = 0;
        for (let i = 0; i < 8; i++) {
            const corner = Points[i];
            const value = readFromGrid(
                x + corner[0],
                y + corner[1],
                z + corner[2]
            );
    
            if (value < threshold) {
                cubeIndex |= 1 << i;
            }
        }
        return TriangleTable[cubeIndex];
    }
    function march_cube(x, y, z) {
        let triangulation = getTriangulation(x, y, z);

        triangulation.forEach(element => {
            if (element === -1) {
                return;
            }
            const point_indices = Edges[element];

            const p1 = Points[point_indices[0]];
            const p2 = Points[point_indices[1]];

            let pos_a = vec3.fromValues(x + p1[0], y + p1[1], z + p1[2]);
            let pos_b = vec3.fromValues(x + p2[0], y + p2[1], z + p2[2]);

            let position = vec3.lerp(pos_a, pos_b, 0.5);

            positions.push(position);
        }); 
    }


    function march_cubes() {
        for (let x = 0; x < gridSize - 1; x++) {
            for (let y = 0; y < gridSize - 1; y++) {
                for (let z = 0; z < gridSize - 1; z++) {
                    march_cube(x, y, z);
                }
            }
        }
    }
    useEffect(() => {
        populateVoxelGrid();
        march_cubes();
        console.log(positions);
       
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

            let maxAbsValue = 0;
            positions.forEach(pos => {
                const [x, y, z] = pos;
                maxAbsValue = Math.max(maxAbsValue, Math.abs(x), Math.abs(y), Math.abs(z));
            });

            // Step 2: Normalize the positions array
            const normalizedPositions = positions.map(pos => {
                const [x, y, z] = pos;
                return [
                    x / maxAbsValue, // Normalize x
                    y / maxAbsValue, // Normalize y
                    z / maxAbsValue, // Normalize z
                ];
            });

            const test = new Float32Array(normalizedPositions.length * 3); // Allocate space for x, y, z for each vertex

            normalizedPositions.forEach((pos, index) => {
                test[index * 3] = pos[0];     // x
                test[index * 3 + 1] = pos[1]; // y
                test[index * 3 + 2] = pos[2]; // z
            });


            const vertices = test;

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
                pass.draw(vertices.length/3) // Draw a triangle
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