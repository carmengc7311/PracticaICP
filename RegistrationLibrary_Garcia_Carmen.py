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
    
    # Filtrar solo los pares válidos
    source_valid = source[valid]
    target_valid = target[nearest_idx[valid]]
    distances = nearest_dist[valid]
    
    # Emparejar puntos y devolverlos en un array (N_valid, 2, D)
    correspondances = np.stack((source_valid, target_valid), axis =1)
    
    return correspondances, distances

def calculate_best_fit_transform(source, target, correspondances):
    #TODO: usar el código de la presentación
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
    
    # Construir la pose 3x3 con la matriz de rotación R y la traslación t
    iteration_transformation = np.eye(3)
    iteration_transformation[0:2,0:2] = R
    iteration_transformation[0:2,2] = t
    return iteration_transformation

    
def transform_points(source_copy, iteration_transformation):
    # TODO: aplicar la transformación
    # método válido solo para 2D
    N = source_copy.shape[0]
    source_copy_h = np.ones((N, 3)) 
    source_copy_h[:, :2] = source_copy
    source_transformed_h = source_copy_h @ iteration_transformation.T
    source_copy = source_transformed_h[:, :2]
    return source_copy

def calculate_rmse(distances):
    #TODO
    return rmse

def icp(target, source,
        max_correspondance_distance = 1000,
        max_iterations = 4,
        metric_delta_threshold = 1e-20):
    src = source.copy()
    prev_metric = float('inf')
    history = []
    dim = source.shape[1]
    total_transformation = np.eye(dim + 1)

    for i in range(max_iterations):
        # Step 1:
        distances, correspondances = calculate_distances_and_correspondences(
            target, src,
            max_correspondance_distance
            )
        
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
