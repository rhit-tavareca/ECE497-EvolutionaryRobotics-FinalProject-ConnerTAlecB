import pyrosim.pyrosim as pyrosim


def World(n,xnum,ynum):
    pyrosim.Start_SDF("boxes.sdf")
    for x in range(xnum):
        for y in range(ynum):
            for i in range(n):
                s = 1 - i/n
                pyrosim.Send_Cube(name="Box", pos=[x,y,0.5 + i] , size=[s,s,s])
    pyrosim.End()

def Robot(num_segments = 10, seg_len = 0.2, seg_w = 0.08, seg_h = 0.08, z_clear = 0.01):
    pyrosim.Start_URDF("robot.urdf")

    z = seg_h/2 + z_clear
    pyrosim.Send_Cube(name="S0" , pos=[0,0,z] , size=[seg_len,seg_w,seg_h])
    

    for i in range(1, num_segments):
        pyrosim.Send_Cube(name=f"S{i}" , pos=[0,0,0] , size=[seg_len,seg_w,seg_h])

    for i in range(num_segments - 1):
        a,b = f"S{i}", f"S{i+1}"
        jx = seg_len / 2
        jpos = [jx, 0, 0]
        pyrosim.Send_Joint(name=f"{a}_{b}", parent=a, child=b, type = "revolute", position=jpos)
       
    pyrosim.End()
Robot()

#World(10,5,5)