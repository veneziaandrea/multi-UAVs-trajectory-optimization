# REGION ASSIGNMENT TO DRONES

def assign_area(drones, voronoi_regions):
    for drone in drones:
        if (voronoi_regions.centroid - drone.drone_start): #2 norm da mettere dopo
            voronoi_regions.id= drone.id
             