import numpy as np
import open3d as o3d
def o3d_to_numpy(geometry_object, attribute_name):
   # Verificar que el atributo existe
    if not hasattr(geometry_object, attribute_name):
        raise AttributeError(
            f"El objeto {type(geometry_object)} no tiene el atributo '{attribute_name}'."
        )

    # Obtener el atributo (Vector3dVector, Vector3iVector, etc.)
    o3d_vector = getattr(geometry_object, attribute_name)

    # Convertir a NumPy
    numpy_array = np.asarray(o3d_vector)

    return numpy_array

def numpy_to_o3d(geometry_object, attribute_name, numpy_array):
    # Asegurar que es un array NumPy
    numpy_array = np.asarray(numpy_array)

    # Detectar tipo de dato
    is_float = numpy_array.dtype.kind in ("f", "d")
    is_int   = numpy_array.dtype.kind in ("i", "u")

    if not (is_float or is_int):
        raise TypeError("El array debe contener enteros o floats.")

    # Detectar dimensiones
    if len(numpy_array.shape) != 2:
        raise ValueError("El array debe tener forma (N, M).")

    num_columns = numpy_array.shape[1]

    # Seleccionar el tipo de Vector adecuado
    if is_float:
        if num_columns == 3:
            vector = o3d.utility.Vector3dVector(numpy_array)
        elif num_columns == 2:
            vector = o3d.utility.Vector2dVector(numpy_array)
        else:
            raise ValueError(f"No existe un VectorXdVector para {num_columns} columnas.")
    else:  # es int
        if num_columns == 3:
            vector = o3d.utility.Vector3iVector(numpy_array)
        elif num_columns == 2:
            vector = o3d.utility.Vector2iVector(numpy_array)
        else:
            raise ValueError(f"No existe un VectorXiVector para {num_columns} columnas.")

    # Asignar el atributo dinámicamente. NO HACE FALTA RETURN
    setattr(geometry_object, attribute_name, vector)
    

def euler_matrix(alpha, beta, gamma):
    # Rotation around X
    Rotx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])

    # Rotation around Y
    Roty = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [ 0,            1, 0           ],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation around Z
    Rotz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [0,              0,             1]
    ])

    # Combined rotation in XYZ sequence: 
    return Rotx @ Roty @ Rotz

# Don't change given method names.
def translate_point_cloud(point_cloud, translation):
    my_points_numpy_array = o3d_to_numpy(point_cloud, "points")
    my_translated_points_array = my_points_numpy_array + translation
    numpy_to_o3d(point_cloud,"points",my_translated_points_array)
    return point_cloud

def rotate_point_cloud(point_cloud, rotation):
    rotx_degrees, roty_degrees, rotz_degrees = rotation
    
    # Numpy only calculates trigonom. functions using radians
    rotx_radians = np.radians(rotx_degrees)
    roty_radians = np.radians(roty_degrees)
    rotz_radians = np.radians(rotz_degrees)
    
    rotation_matrix  = euler_matrix(rotx_radians, roty_radians, rotz_radians)
    my_rot_points_numpy_array = o3d_to_numpy(point_cloud,"points")
    
    centroid = np.mean(my_rot_points_numpy_array, axis=0)
    
    # 1. Translate to origin
    origin_points_numpy_array = my_rot_points_numpy_array - centroid
    
    # 2. Rotate
    rotated_origin_numpy_points = origin_points_numpy_array @ rotation_matrix
    
    # 3. Translate back 
    rotated_numpy_points = rotated_origin_numpy_points + centroid
    
    numpy_to_o3d(point_cloud,"points",rotated_numpy_points)
    return point_cloud

def color_point_cloud(point_cloud, color):
    # Convertimos al rango que usa Open3d: [0,1]
    clipped_color = color.clip(min=0,max=1)
    # Obtenemos puntos solo para saber N
    points_numpy_array = o3d_to_numpy(point_cloud,"points")
    # Creamos array Nx3 con el color repetido
    point_colors = np.tile(clipped_color,(points_numpy_array.shape[0],1))
    # asignar colores sin tocar las coordenadas
    numpy_to_o3d(point_cloud,"colors",point_colors)
    return point_cloud
    
def color_point_cloud_with_density(point_cloud, distance, min_density, max_density):
    
    return point_cloud

def translate_mesh(mesh, translation):
    return mesh

def rotate_mesh(mesh, rotation):
    return mesh  

def color_mesh(mesh, color):
    return mesh

def compute_mesh_vertex_normals(mesh):
    return mesh