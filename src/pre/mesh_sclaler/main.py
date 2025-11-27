import meshio

def rescale_msh(input_path, output_path, scale_factor):
    mesh = meshio.read(input_path)
    mesh.points *= scale_factor
    meshio.write(output_path, mesh, file_format="gmsh22", binary=False)

if __name__ == "__main__":
    i_path = "../../../mesh/3D/sampleA_ideal/sampleA_ideal.msh"
    o_path = "../../../mesh/3D/sampleA_ideal/sampleA_ideal_scaled.msh"
    rescale_msh(i_path, o_path, 0.001)