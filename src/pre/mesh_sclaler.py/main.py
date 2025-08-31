import meshio

def rescale_msh(input_path, output_path, scale_factor):
    mesh = meshio.read(input_path)
    mesh.points *= scale_factor
    meshio.write(output_path, mesh, file_format="gmsh22", binary=False)

if __name__ == "__main__":
    i_path = "../../mesh/2D/bend_tube/bend_tube_extended.msh"
    o_path = "../../mesh/2D/bend_tube/bend_tube_extended_scaled.msh"
    rescale_msh(i_path, o_path, 0.001)