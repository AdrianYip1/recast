Adding an object to an existing photo relies on retouching through software like Photoshop or using GenAI. Editing software requires prerequisite skills and is time consuming while using GenAI often removes details in the photos, produces inaccurate lighting and shadows, and leaves users unable to fine tune how they want their final results to look manually. 

ReCast rebuilds the scene photo as a 3D space, recovers its lighting, and extracts the object from a secondary photo and places it into the scene photo. A real time **Vulkan** path tracer calculates the lighting of the newly added object given the 3D scene’s illumination and renders its shadows and reflections onto the original photo, which can then be relit freely.

## Part I: Rebuilding the Scene

To start, upload a scene image, which we use to reconstruct a 3D space the renderer can work with. Phone lenses distort straight edges, so the photo is first undistorted via the **lensfun** camera database. The straight edges in the room are then grouped by **RANSAC** into three vanishing points, which give the camera's focal length, and rotation, the same method Hartley and Zisserman describe and fSpy uses.

We use two locally-ran pretrained models: **OneFormer (ADE20K panoptic)** labels every region of the photo as floor, wall, or a named piece of furniture, and **Depth Anything V2** finds the relative depth of every pixel. Since the solved camera knows exactly how far away the floor is, fitting the depth on floor pixels turns the pieces of furniture into proxies anchored to the ground.

Those proxies were AABB initially, which left artifacts and box-like margins around round chairs and cylindrical tables and caused objects to be positioned inaccurately when moved around the scene. We fixed this using a convex hull around its 3D points, keeping only the voxels that fall inside the furniture's cutout and behind the visible surface.

## Part II: The Object

The object is rebuilt from our other image into a **glTF mesh**. The user drags an AABB around this image inside ReCast, and **SAM (ViT B)** uses that box to cut the object out with a soft alpha to keeps its fine edges. 

This cutout goes to **TripoSR**, which predicts a triplane and a density field from the image. **Marching cubes** convert this field into a usable glTF mesh, where **xatlas** unwraps it, and texture is baked from the photo wherever the surface faced the camera. Lighting of this object is removed removed so the Vulkan engine can relight it: an order 1 spherical harmonics fit (Ramamoorthi and Hanrahan) divides out the shading and the colour of that light. Finally the mesh is stood upright, scaled to metres, and exported to glTF. Every object is stored in a cache, speeding up performance of the renderer.

## Part III: Lighting and Path Tracing

ReCast renders in a custom Vulkan 1.3 engine with dynamic rendering, a glTF loader, and **Dear ImGui**. The lights are recovered from the photo itself, where strengths and ambient lights are fitted by least squares to the photo's shading. To automatically calibrate lights, a search over a scene's bright spots determines where **point lights** and the **sun (directional light)** are.

The object is then path traced in a compute shader using hardware ray queries (VK_KHR_ray_query). The scene itself is never relit, only the object as the motivation of ReCast is to drop an object into a preexisting scene in a realistic manner. Following differential rendering (Debevec, Karsch), each room's pixel is found from the photo multiplied by how much the object changes the light arriving there. Our renderer included features like: denoising, caustics, GGX gloss, reflections, and refractions.

Not every GPU is able to run this to the same capacity. Integrated graphics couldn't bind the thousand textures the engine expected, and some GPUs have no ray tracing at all, so the engine falls back to fewer textures and to a raster renderer with shadow maps.

## Part IV: Results, Motivations & Future Plans

Asking a gen AI model to add an object means it regenerates the whole image. Finer details like text, textures, and faces drift or blur, the lighting and shadows often just look plausible rather than being physically correct, and when something is off, there is no definite way to fix just that part. The only option is to change the prompt and hope the next result is closer. Photoshop gives full control, however, it asks a lot of a beginner; cutting the object out, matching the perspective, and painting shadows and reflections by hand takes skill, practice, and hours of work.

ReCast aims to be both a precise tool for this situation while being simple to use. The original scene is never regenerated, only the pixels the object actually affects change, so every detail of the scene stays exactly as it was taken. Adding an object only takes a few clicks: choose the scene photo, object photo, drag a box around the object, and drag it into place. The path tracer works out its lighting, shadows, and reflections from the scene itself.

Everything can still be adjusted by hand (resizing, rotations, translations) with a user-friendly interface. The original scene photo can be shown at any time to compare, and the finished photo is exported at full resolution with no interface on it.

Looking ahead, we envision Recast as a plugin for professional photo editors like Photoshop, where it would bridge the gap between novices and experts and give beginners the confidence to discover the unknown.
## Built with

- **[enginemath](https://github.com/AdrianYip1/enginemath.git)**, our custom math library for vector and matrix math
- **[Vulkan 1.3](https://www.vulkan.org/)** for rendering and path tracing (dynamic rendering, `VK_KHR_ray_query`), with **[GLFW](https://github.com/glfw/glfw)** (windowing and input), **[Dear ImGui](https://github.com/ocornut/imgui)** (UI), and **[stb_image and stb_image_write](https://github.com/nothings/stb)** (image loading and export)
- **[cgltf](https://github.com/jkuhlmann/cgltf)** for glTF model loading
- **[nlohmann/json](https://github.com/nlohmann/json)** for JSON parsing
- **[Python](https://www.python.org/)** with **[PyTorch](https://pytorch.org/)** and **[Hugging Face Transformers](https://huggingface.co/docs/transformers)** for running our models locally
- **[OpenCV](https://opencv.org/)**, **[NumPy](https://numpy.org/)**, **[SciPy](https://scipy.org/)**, and **[scikit-image](https://scikit-image.org/)** for the camera solve, depth fitting, and space carving
- **[trimesh](https://github.com/mikedh/trimesh)** and **[xatlas](https://github.com/jpcy/xatlas)** for building, cleaning, and unwrapping meshes
- **[lensfun](https://lensfun.github.io/)** (through [lensfunpy](https://github.com/letmaik/lensfunpy)) for lens distortion correction
- **[Pillow](https://python-pillow.org/)** for photo loading

## Credits

- [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) by Jain et al. (CVPR 2023)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) by Yang et al. (NeurIPS 2024)
- [Segment Anything](https://arxiv.org/abs/2304.02643) by Kirillov et al. (ICCV 2023)
- [TripoSR: Fast 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/2403.02151) by Tochilkin et al. (2024)
- [Rendering Synthetic Objects into Real Scenes](https://doi.org/10.1145/280814.280864) by Debevec (SIGGRAPH 1998)
- [Rendering Synthetic Objects into Legacy Photographs](https://doi.org/10.1145/2024156.2024191) by Karsch, Hedau, Forsyth, Hoiem (SIGGRAPH Asia 2011)
- [An Efficient Representation for Irradiance Environment Maps](https://doi.org/10.1145/383259.383317) by Ramamoorthi, Hanrahan (SIGGRAPH 2001)
- [Spatiotemporal Variance-Guided Filtering](https://doi.org/10.1145/3105762.3105770) by Schied et al. (HPG 2017)
- [Fast Noise Variance Estimation](https://doi.org/10.1006/cviu.1996.0060) by Immerkær (CVIU 1996)
- [Multiple View Geometry in Computer Vision](https://doi.org/10.1017/CBO9780511811685) by Hartley, Zisserman
- Glass dragon (DragonAttenuation) from the [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), based on the [Stanford Dragon](https://graphics.stanford.edu/data/3Dscanrep/)
- [Montserrat](https://fonts.google.com/specimen/Montserrat) and [Reenie Beanie](https://fonts.google.com/specimen/Reenie+Beanie) fonts (SIL Open Font License)

  ## Built for
  Hack Western 13
