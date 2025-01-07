import React, { useEffect, useRef } from 'react';
import { mat3, mat4, vec3, vec4 } from 'wgpu-matrix';
import { createNoise3D } from 'simplex-noise'
import { EdgeMasks, EdgeVertexIndices, TriangleTable } from './marchingCubesLookup';


const WebGPUCanvas = () => {
    const canvasRef = useRef(null);

    //Grid dimenstions
    const gridSize = 16;
    const threshold = 0.5;

    //functon to generate a 3D grid of random values
    function generateGrid3D(gridSize) {
        const grid = new Float32Array(gridSize * gridSize * gridSize);
        const noise = createNoise3D();
        for (let z = 0; z < gridSize; z++) {
            for (let y = 0; y < gridSize; y++) {
                for (let x = 0; x < gridSize; x++) {
                    const value = noise(x * 0.1, y * 0.1, z * 0.1);
                    const index = x + y * gridSize + z * gridSize * gridSize;
                    grid[index] = value;
                }
            }
        }
        return grid;
    }

    function flattenTriangles(triangles) {
        const flattened = [];
        for (const triangle of triangles) {
            if (triangle) {
                flattened.push(triangle[0]);
                flattened.push(triangle[1]);
                flattened.push(triangle[2]);
            }
            
        }
        console.log(flattened);
        return new Float32Array(flattened);
    }

    // Fixed interpolation function
    function interpolateVertex(v1, v2, val1, val2, threshold) {
        if (Math.abs(threshold - val1) < 0.00001) return v1;
        if (Math.abs(threshold - val2) < 0.00001) return v2;
        if (Math.abs(val1 - val2) < 0.00001) return v1;
        
        const t = (threshold - val1) / (val2 - val1);
        return [
            v1[0] + t * (v2[0] - v1[0]),
            v1[1] + t * (v2[1] - v1[1]),
            v1[2] + t * (v2[2] - v1[2])
        ];
    }

    function normalizeVertices(vertices, targetMin = -0.5, targetMax = 0.5) {
        // Find the min and max values of the vertex coordinates
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
    
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
    
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        }
    
        // Calculate the range of the vertex coordinates
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const rangeZ = maxZ - minZ;
    
        // Calculate the scale factor to fit within the target range
        const scaleX = (targetMax - targetMin) / rangeX;
        const scaleY = (targetMax - targetMin) / rangeY;
        const scaleZ = (targetMax - targetMin) / rangeZ;
    
        // Normalize the vertices
        const normalizedVertices = new Float32Array(vertices.length);
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
    
            // Scale and translate to the target range
            normalizedVertices[i] = (x - minX) * scaleX + targetMin;
            normalizedVertices[i + 1] = (y - minY) * scaleY + targetMin;
            normalizedVertices[i + 2] = (z - minZ) * scaleZ + targetMin;
        }
    
        return normalizedVertices;
    }

    useEffect(() => {
        const grid = generateGrid3D(gridSize);
        const allVertices = []; // Global list of vertices
        const allIndices = []; // Global list of indices

        // Main marching cubes loop
        for (let z = 0; z < gridSize - 1; z++) {
            for (let y = 0; y < gridSize - 1; y++) {
                for (let x = 0; x < gridSize - 1; x++) {
                    // Get the values at the cube's corners
                    const cube = [
                        grid[x + y * gridSize + z * gridSize * gridSize],
                        grid[(x + 1) + y * gridSize + z * gridSize * gridSize],
                        grid[(x + 1) + (y + 1) * gridSize + z * gridSize * gridSize],
                        grid[x + (y + 1) * gridSize + z * gridSize * gridSize],
                        grid[x + y * gridSize + (z + 1) * gridSize * gridSize],
                        grid[(x + 1) + y * gridSize + (z + 1) * gridSize * gridSize],
                        grid[(x + 1) + (y + 1) * gridSize + (z + 1) * gridSize * gridSize],
                        grid[x + (y + 1) * gridSize + (z + 1) * gridSize * gridSize],
                    ];

                    let cubeIndex = 0;
                    for (let i = 0; i < 8; i++) {
                        if (cube[i] < threshold) cubeIndex |= (1 << i);
                    }

                    const edgeMask = EdgeMasks[cubeIndex];
                    if (edgeMask === 0) continue;

                    const vertexList = []; // Local list of vertices for this cube
                    for (let i = 0; i < 12; i++) {
                        if (edgeMask & (1 << i)) {
                            const edge = EdgeVertexIndices[i];
                            const p1 = edge[0];
                            const p2 = edge[1];

                            // Create vertices as arrays instead of vec3
                            const v1 = [
                                x + (p1[0] === 1 ? 1 : 0),
                                y + (p1[1] === 1 ? 1 : 0),
                                z + (p1[2] === 1 ? 1 : 0),
                            ];
                            const v2 = [
                                x + (p2[0] === 1 ? 1 : 0),
                                y + (p2[1] === 1 ? 1 : 0),
                                z + (p2[2] === 1 ? 1 : 0),
                            ];

                            // Get correct indices for cube values
                            const val1 = cube[edge[0]]; // Use the first vertex index of the edge
                            const val2 = cube[edge[1]]; // Use the second vertex index of the edge

                            const vertex = interpolateVertex(v1, v2, val1, val2, threshold);
                            vertexList.push(vertex);
                        }
                    }

                    // Create triangles using the triangle table
                    const triangleIndices = TriangleTable[cubeIndex];
                    for (let i = 0; i < triangleIndices.length && triangleIndices[i] !== -1; i += 3) {
                        // Add vertices to the global list
                        const index1 = allVertices.length;
                        allVertices.push(vertexList[triangleIndices[i]]);
                        allVertices.push(vertexList[triangleIndices[i + 1]]);
                        allVertices.push(vertexList[triangleIndices[i + 2]]);

                        // Add indices to the global list
                        allIndices.push(index1, index1 + 1, index1 + 2);
                    }
                }
            }
        }

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

            // const vertices = new Float32Array([
            //     // X, Y, Z coordinates
            //     -0.5, -0.5, -0.5,   // Vertex 0
            //      0.5, -0.5, -0.5,   // Vertex 1
            //      0.5,  0.5, -0.5,   // Vertex 2
            //     -0.5,  0.5, -0.5,   // Vertex 3
            //     -0.5, -0.5,  0.5,   // Vertex 4
            //      0.5, -0.5,  0.5,   // Vertex 5
            //      0.5,  0.5,  0.5,   // Vertex 6
            //     -0.5,  0.5,  0.5    // Vertex 7
            // ]);

            const flattened = flattenTriangles(allVertices);
            const vertices = normalizeVertices(flattened);

            const indices = new Uint32Array(allIndices);
        
            // const indices = new Uint32Array([
            //     // Front face
            //     0, 1, 2,  2, 3, 0,
            //     // Back face
            //     4, 5, 6,  6, 7, 4,
            //     // Left face
            //     0, 4, 7,  7, 3, 0,
            //     // Right face
            //     1, 5, 6,  6, 2, 1,
            //     // Top face
            //     3, 7, 6,  6, 2, 3,
            //     // Bottom face
            //     0, 1, 5,  5, 4, 0
            // ]);

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
                pass.drawIndexed(indices.length); // Draw a triangle
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