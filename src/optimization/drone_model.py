def drone_model(pos, vel, acc, dt): 
    pos_nxt = pos + vel*dt + 0.5*acc*dt**2
    vel_nxt = vel + acc*dt
    acc_nxt = acc
    return pos_nxt, vel_nxt