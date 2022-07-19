import bpy

obj_abspath = bpy.path.abspath('//scene.obj')
mtl_abspath = bpy.path.abspath('//scene.mtl')
# bpy.ops.export_scene.obj(filepath=obj_abspath, use_triangles=True, \
#                             axis_forward='Y', axis_up='Z')
                            
for img in bpy.data.images:
    abspath = bpy.path.abspath(img.filepath).replace('.dds', '.jpg')
    try:
        img.scale(512, 512)
        img.save_render(filepath=abspath)
    except:
        print(abspath)
        continue
    
mtl_file = open(mtl_abspath, 'r')
mtl_file_lines = []
for line in mtl_file:
    mtl_file_lines.append(line.replace('.dds', '.jpg'))
    
mtl_file.close()

mtl_file = open(mtl_abspath, 'w')
mtl_file.writelines(mtl_file_lines)
mtl_file.close()