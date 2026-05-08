import numpy as np

def calculate_distances_and_correspondences(
        target, source,
        max_correspondance_distance):

    # source : (N, D)
    # target : (M, D)
    # Matriz de diferencias NxMxD
    diff = source[:, None, :] - target[None, :, :]
    
    # Matriz de distancias NxM
    dists = np.linalg.norm(diff, axis=2)
    
    # Índice del target más cercano para cada punto del source
    nearest_idx = dists.argmin(axis=1)
    
    # Distancia correspondiente
    nearest_dist = dists[np.arange(len(source)), nearest_idx]
    
    # "Máscara" de correspondencias válidas (puntos dentro del rango)
    valid = nearest_dist <= max_correspondance_distance
    
    # Si no hay correspondencias válidas devolvemos arrays vacíos para imple-
    # mentar el break del icp
    if not np.any(valid):
        D = source.shape[1]
        return np.empty((0, 2, D)), np.empty((0,))
    
    # Filtrar solo los pares válidos
    source_valid = source[valid]
    target_valid = target[nearest_idx[valid]]
    distances = nearest_dist[valid]
    
    # Emparejar puntos y devolverlos en un array (N_valid, 2, D)
    correspondances = np.stack((source_valid, target_valid), axis =1)
    
    return correspondances, distances

def calculate_best_fit_transform(source, target, correspondances):
    # Calcula la transformación rígida óptima entre dos sets de puntos en N dimensiones.
    # Devuelve una matriz de transformación de dimensiones (N+1)x(N+1)
    
    # Seleccionar puntos que tienen correspondencia
    source_correspondances = correspondances[:, 0] 
    target_correspondances = correspondances[:, 1]
    
    # Calcular centroides de dichos puntos
    centroid_source = source_correspondances.mean(axis=0)
    centroid_target = target_correspondances.mean(axis=0)
    
    # Centrar los puntos con correspondencia
    source_centered = source_correspondances - centroid_source
    target_centered = target_correspondances - centroid_target
    
    # Calcular la matriz de covarianza
    H =  source_centered.T @ target_centered
    
    # Descomposición en valores singulares
    U,_,Vt = np.linalg.svd(H) # no necesito guardar los valores singulares
    
    # Cálculo de la matriz de rotación
    R = Vt.T @ U.T
    
    # Corrección del reflejo: si rotación en realidad es un reflejo, lo convertimos en una rotación
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # Cálculo de la traslación
    t = centroid_target - R @ centroid_source
    
    # Construir la matriz de transformación (3x3 o 4x4)
    # con la matriz de rotación R y la traslación t
    D = source_correspondances.shape[1] # dimensión: 2 o 3
    
    iteration_transformation = np.eye(D+1)
    iteration_transformation[0:D,0:D] = R
    iteration_transformation[0:D,D] = t
    return iteration_transformation

    
def transform_points(source_copy, iteration_transformation):
    
    N = source_copy.shape[0] # número de puntos
    D = source_copy.shape[1] # dimensión: 2 o 3
    source_copy_h = np.ones((N, D + 1)) 
    source_copy_h[:, :D] = source_copy
    source_transformed_h = source_copy_h @ iteration_transformation.T
    source_copy = source_transformed_h[:, :D]
    return source_copy

def calculate_rmse(distances):
    #TODO: df
    if len(distances) == 0:
        return np.inf 
    
    dist_squared = distances **2
    dist_squared_mean = dist_squared.mean()
    rmse = np.sqrt(dist_squared_mean) #raíz cuadrada de la media
    return rmse

def icp(target, source,
        max_correspondance_distance = 1000,
        max_iterations = 100,
        metric_delta_threshold = 1e-7):
    src = source.copy()
    prev_metric = float('inf')
    history = []
    dim = source.shape[1]
    total_transformation = np.eye(dim + 1)

    for i in range(max_iterations):
        # Step 1:
        correspondances, distances = calculate_distances_and_correspondences(
            target, src,
            max_correspondance_distance
            )
        # Si ya no hay correspondencias válidas significa que no podemos mejo-
        # rar el ICP -- lo paramos:
        if len(distances) == 0:
            break
 
        # Step 2:
        iteration_transformation = calculate_best_fit_transform(src, target, correspondances)
        
        # Step 3:
        total_transformation = iteration_transformation @ total_transformation
        
        # Step 4:
        src = transform_points(src, iteration_transformation)
        
        # Step 5:
        metric = calculate_rmse(distances)
        
        history.append((metric, total_transformation))
        if abs(prev_metric - metric) < metric_delta_threshold:
            break
        prev_metric = metric
    
    return total_transformation, history
