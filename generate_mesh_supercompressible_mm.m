%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%  Generate input files for FEA of the RVEs with elliptical particles     %
%                                                                         %
%  This function is called by the code: ICME_main_code.m                  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    Miguel A. Bessa - M.A.Bessa@tudelft.nl               %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function generate_mesh_supercompressible_mm(DoE_points,Input_points,UserSettings,jDoE,kImperfection,iInput,jDoE_dir)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Python File for Abaqus CAE                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load_dof = Input_points(iInput,1);

% Convert DoE_points and Input_points into cell arrays to assign variable
% names in one line afterward:
DoE_point = num2cell(DoE_points(jDoE,:));
Input_point = num2cell(Input_points(iInput,:));

% Assign variables in one liners:
[ A_D1 , G_E , Ix_D1 , Iy_D1 , J_D1 , P_D , ConeSlope ] = DoE_point{:};
% [P_D , nStories, MastDiameter, ConeSlope , Batten_D_Longeron_b ] = DoE_point{:};
% [MastDiameter, P_D , nStories, ConeSlope ] = DoE_point{:};
% 
[VertexPolygon, nStories, BattenDiamater_d, power, pinned_joints, ...
    EModulus, D1, BattenFormat, BattenMaterial, ...
    DiagonalRadius, DiagonalMaterial, ...
    Twist_angle, transition_length_ratio] = Input_point{:};

% E = 1826; % Hard coded Young's modulus (Longerons & Battens)
% nu = 0.27; % Hard coded Poisson ratio (Longerons & Battens)
% G = E/(2*(1+nu))
G = EModulus * G_E; %The Shear Modulus (Longerons)
nu = (EModulus/(2*G))-1; %The Poisson ratio (Longerons)


P = P_D*D1; % Pitch

% d = d_P * P; % diameter of longeron cross-section
% d = d_D1 * D1; % diameter of longeron cross-section
% Longeron_b = (EI_l*4/(pi*E))^(1/4); % Radius of longeron cross-section
% I = (d/2)^4*pi/4; % Longeron moment of inertia (in x and y, as it has circular cross-section)

% E = EI/I; % Compute the Young's modulus of the material

MastDiameter = D1; % Mast diameter
MastPitch = P; % Mast pitch
% Longeron_b = d/2.0; % Radius of longeron cross-section

%%%%%%%% Added by Piotr
Longeron_CS = A_D1*D1^2;
Ix = Ix_D1*D1^4;
Iy = Iy_D1*D1^4;
J = J_D1*D1^4;

% J_l = pi/2*Longeron_b^4;
% G = GJ_l/J_l;
% nu = E/(2*G)-1;
% BattenRadius = (GJ_b*2/(pi*G))^(1/4); % Radius of batten cross-section
% BattenRadius = Ratio_BattenD_d*d/2.0; % Radius of batten cross-section

% [VertexPolygon,power,Longeron_b,LongeronRatio_a_b,...
%     BattenRadius,DiagonalRadius,Twist_angle,transition_length_ratio] = Input_point{:};
% pinned_joints = 1; LongeronMaterial=1; BattenFormat=0; BattenMaterial=1; DiagonalMaterial=1;

% MastPitch = P_D*MastDiameter;

% Batten_D = Batten_D_Longeron_b*Longeron_b;
% Batten_D = BattenRadius*2.0;

% VertexPolygon = Input_points(iInput,1);
% % VertexPolygon = DoE_points(jDoE,1);
% 
% power = Input_points(iInput,2);
% % power = DoE_points(jDoE,2);
% 
% % MastDiameter = Input_points(iInput,3);
% MastDiameter = DoE_points(jDoE,1);
% 
% nStories = Input_points(iInput,3);
% % nStories = DoE_points(jDoE,3);
% 
% MastPitch = Input_points(iInput,4)*MastDiameter;
% % MastPitch = DoE_points(jDoE,2)*MastDiameter;
% 
% pinned_joints = Input_points(iInput,5);
% % pinned_joints = DoE_points(iInput,5);
% 
% % Longeron_b = Input_points(iInput,3);
% Longeron_b = DoE_points(jDoE,2);
% 
% % LongeronRatio_a_b = Input_points(iInput,4);
% LongeronRatio_a_b = DoE_points(jDoE,3);
% 
% LongeronMaterial = Input_points(iInput,6);
% % LongeronMaterial = DoE_points(jDoE,3);
% 
% % Batten_D = Input_points(iInput,5)*Longeron_a;
% Batten_D = DoE_points(jDoE,4)*Longeron_b;
% 
% BattenFormat = Input_points(iInput,7);
% % BattenFormat = DoE_points(iInput,5);
% 
% BattenMaterial = Input_points(iInput,8);
% % BattenMaterial = DoE_points(iInput,6);
% 
% DiagonalRadius = Input_points(iInput,9);
% % DiagonalRadius = DoE_points(jDoE,9);
% 
% DiagonalMaterial = Input_points(iInput,10);
% % DiagonalRadius = DoE_points(jDoE,9);
% 
% ConeSlope = Input_points(iInput,11);
% % ConeSlope = DoE_points(jDoE,6);
% 
% Twist_angle = Input_points(iInput,12);
% % Twist_angle = DoE_points(jDoE,11);
% 
% transition_length_ratio = Input_points(iInput,13);
% % transition_length_ratio = DoE_points(jDoE,12);
%
%
% disp(' ');
% disp('Started creation of Python file');
%
scriptfile_name = strcat(jDoE_dir,'/script_DoE',num2str(jDoE),'_meshing.py');
fid = fopen(scriptfile_name,'wt');
%
% Heading
%
fprintf(fid,'# Abaqus/CAE script\n');
fprintf(fid,'# Created by M.A. Bessa (M.A.Bessa@tudelft.nl) on %s\n',datestr(now));
fprintf(fid,'#\n');
fprintf(fid,'from abaqus import *\n');
fprintf(fid,'from abaqusConstants import *\n');
fprintf(fid,'session.viewports[''Viewport: 1''].makeCurrent()\n');
fprintf(fid,'#session.viewports[''Viewport: 1''].maximize()\n');
fprintf(fid,'from caeModules import *\n');
fprintf(fid,'from driverUtils import executeOnCaeStartup\n');
fprintf(fid,'executeOnCaeStartup()\n');
fprintf(fid,'Mdb()\n');
fprintf(fid,'#\n');
fprintf(fid,'import numpy\n');
fprintf(fid,'\n');
fprintf(fid,'#------------------------------------------------------------\n');
% A new model database has been created.
% The model "Model-1" has been created.
fprintf(fid,'os.chdir(r''%s'')\n',jDoE_dir);
fprintf(fid,'#\n');
fprintf(fid,'#-------------------------------------------------------------\n');
fprintf(fid,'# Parameters:\n');
fprintf(fid,'VertexPolygon = %i # Number of vertices (sides) of the polygon base\n',VertexPolygon);
fprintf(fid,'power = %7.5e # Power law exponent establishing the evolution of the spacing between battens\n',power);
fprintf(fid,'MastDiameter = %7.5e # Radius of the circumscribing circle of the polygon\n',MastDiameter);
fprintf(fid,'nStories = %i # Number of stories in HALF of the strut (i.e. in a single AstroMast!)\n',nStories);
fprintf(fid,'MastPitch = %7.5e # Pitch length of the strut (i.e. a single AstroMast!)\n',MastPitch);
fprintf(fid,'pinned_joints = %i # (1 = batten are pinned to longerons, 0 = battens and longerons are a solid piece)\n',pinned_joints);
% fprintf(fid,'Longeron_b = %7.5e # (smallest cross-section length in rectangular profile, or radius in circ. profile)\n',Longeron_b);
% fprintf(fid,'LongeronRatio_a_b = %7.5e # (a/b ratio, i.e. ratio between largest and smallest side of rect. profile)\n',LongeronRatio_a_b);
fprintf(fid,'Longeron_CS = %7.5e # (Cross Section of the longeron)\n',Longeron_CS);
fprintf(fid,'Ix = %7.5e # (Second moment of area around X axis )\n',Ix);
fprintf(fid,'Iy = %7.5e # (Second moment of area around Y axis )\n',Iy);
fprintf(fid,'J = %7.5e # (Second moment of area around X axis )\n',J);
fprintf(fid,'Emodulus = %7.5e # (Youngus Modulus)\n',EModulus);
fprintf(fid,'Gmodulus = %7.5e # (Shear Modulus)\n',G);
fprintf(fid,'nu = %7.5e # (Poisson Ratio)\n',nu);
% fprintf(fid,'LongeronMaterial = %i # (1 = NiTi alloy, 2 = PC , 3 = PLA)\n',LongeronMaterial);
% fprintf(fid,'BattenRadius = %7.5e\n',BattenRadius);
% fprintf(fid,'BattenFormat = %i # (0 = closed regular polygon linking longerons, 1 = star shaped)\n',BattenFormat);
% fprintf(fid,'BattenMaterial = %i # (1 = NiTi alloy, 2 = PC , 3 = PLA)\n',BattenMaterial);
% fprintf(fid,'DiagonalRadius = %7.5e # if value is 0.0 then no diagonals will be included\n',DiagonalRadius);
% fprintf(fid,'DiagonalMaterial = %i # (1 = NiTi alloy, 2 = PC , 3 = PLA)\n',DiagonalMaterial);
fprintf(fid,'ConeSlope = %7.5e # Slope of the longerons (0 = straight, <0 larger at the top, >0 larger at the bottom)\n',ConeSlope);
fprintf(fid,'Twist_angle = %7.5e # Do you want to twist the longerons?\n',Twist_angle);
fprintf(fid,'transition_length_ratio = %7.5e # Transition zone for the longerons\n',transition_length_ratio);
fprintf(fid,'#------------------------------------------------------------\n');
fprintf(fid,'\n');
% fprintf(fid,'Longeron_a = LongeronRatio_a_b*Longeron_b\n');
fprintf(fid,'MastRadius = MastDiameter/2.0\n');
fprintf(fid,'MastHeight = nStories*MastPitch\n');
fprintf(fid,'\n');
% fprintf(fid,'Mesh_size = min(MastRadius,MastPitch)/10.0\n');
fprintf(fid,'Mesh_size = min(MastRadius,MastPitch)/300.0\n');
fprintf(fid,'\n');
fprintf(fid,'session.viewports[''Viewport: 1''].setValues(displayedObject=None)\n');
fprintf(fid,'\n');
% fprintf(fid,'# Select Material for Longerons\n');
% fprintf(fid,'if LongeronMaterial == 1:\n');
% fprintf(fid,'    Longeron_Material = ''NiTi_alloy''\n');
% fprintf(fid,'elif LongeronMaterial == 2:\n');
% fprintf(fid,'    Longeron_Material = ''PC''\n');
% fprintf(fid,'elif BattenMaterial == 3:\n');
% fprintf(fid,'    Longeron_Material = ''PLA''\n');
% fprintf(fid,'\n');
% fprintf(fid,'# Select Material for Battens\n');
% fprintf(fid,'if BattenMaterial == 1:\n');
% fprintf(fid,'    Batten_Material = ''NiTi_alloy''\n');
% fprintf(fid,'elif BattenMaterial == 2:\n');
% fprintf(fid,'    Batten_Material = ''PC''\n');
% fprintf(fid,'elif BattenMaterial == 3:\n');
% fprintf(fid,'    Batten_Material = ''PLA''\n');
% fprintf(fid,'\n');
% fprintf(fid,'# Select Material for Diagonals\n');
% fprintf(fid,'if DiagonalMaterial == 1:\n');
% fprintf(fid,'    Diagonal_Material = ''NiTi_alloy''\n');
% fprintf(fid,'elif DiagonalMaterial == 2:\n');
% fprintf(fid,'    Diagonal_Material = ''PC''\n');
% fprintf(fid,'elif DiagonalMaterial == 3:\n');
% fprintf(fid,'    Diagonal_Material = ''PLA''\n');
% fprintf(fid,'\n');

fprintf(fid,'\n');
fprintf(fid,'# Create all the joints of the a single Deployable Mast:\n');
fprintf(fid,'joints = numpy.zeros((nStories+1,VertexPolygon,3))\n');
fprintf(fid,'joints_outter = numpy.zeros((nStories+1,VertexPolygon,3))\n');
fprintf(fid,'for iStorey in range(0,nStories+1,1):\n');
fprintf(fid,'    for iVertex in range(0,VertexPolygon,1):\n');
fprintf(fid,'        # Constant spacing between each storey (linear evolution):\n');
fprintf(fid,'        Zcoord = MastHeight/nStories*iStorey\n');
fprintf(fid,'        # Power-law spacing between each storey (more frequent at the fixed end):\n');
fprintf(fid,'        #        Zcoord = MastHeight*(float(iStorey)/float(nStories))**power\n');
fprintf(fid,'        # Power-law spacing between each storey (more frequent at the rotating end):\n');
fprintf(fid,'        #        Zcoord = -MastHeight/(float(nStories)**power)*(float(nStories-iStorey)**power)+MastHeight\n');
fprintf(fid,'        # Exponential spacing between each storey\n');
fprintf(fid,'        #        Zcoord =(MastHeight+1.0)/exp(float(nStories))*exp(float(iStorey))\n');
fprintf(fid,'        #\n');
fprintf(fid,'        Xcoord = MastRadius*cos(2.0*pi/VertexPolygon*iVertex + Twist_angle*min(Zcoord/MastHeight/transition_length_ratio,1.0))\n');
fprintf(fid,'        Ycoord = MastRadius*sin(2.0*pi/VertexPolygon*iVertex + Twist_angle*min(Zcoord/MastHeight/transition_length_ratio,1.0))\n');
fprintf(fid,'        # Save point defining this joint:\n');
fprintf(fid,'        joints[iStorey,iVertex,:] = (Xcoord*(1.0-min(Zcoord,transition_length_ratio*MastHeight)/MastHeight*ConeSlope),Ycoord*(1.0-min(Zcoord,transition_length_ratio*MastHeight)/MastHeight*ConeSlope),Zcoord)\n');
fprintf(fid,'        #\n');
fprintf(fid,'        center = (0.0,0.0)\n');
fprintf(fid,'        vec = joints[iStorey,iVertex,0:2]-center\n');
fprintf(fid,'        norm_vec = numpy.linalg.norm(vec)\n');
fprintf(fid,'        joints_outter[iStorey,iVertex,2] = joints[iStorey,iVertex,2]\n');
fprintf(fid,'        joints_outter[iStorey,iVertex,0:2] = joints[iStorey,iVertex,0:2]\n'); %+ vec/norm_vec*Longeron_a\n'); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(fid,'    # end iSide loop\n');
fprintf(fid,'    \n');
fprintf(fid,'#end iStorey loop\n');
fprintf(fid,'\n');
fprintf(fid,'# Create the longerons:\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons = mdb.models[''Model-1''].Part(name=''longerons'', dimensionality=THREE_D, \n');
fprintf(fid,'    type=DEFORMABLE_BODY)\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons = mdb.models[''Model-1''].parts[''longerons'']\n');
fprintf(fid,'session.viewports[''Viewport: 1''].setValues(displayedObject=p_longerons)\n');
fprintf(fid,'\n');
fprintf(fid,'d_longerons, r_longerons = p_longerons.datums, p_longerons.referencePoints\n');
fprintf(fid,'\n');
fprintf(fid,'LocalDatum_list = [] # List with local coordinate system for each longeron\n');
fprintf(fid,'long_midpoints = [] # List with midpoints of longerons (just to determine a set containing the longerons)\n');
fprintf(fid,'e_long = p_longerons.edges\n');
fprintf(fid,'\n');
fprintf(fid,'\n');
fprintf(fid,'for iVertex in range(0,VertexPolygon,1):\n');
fprintf(fid,'    # First create local coordinate system (useful for future constraints, etc.):\n');
fprintf(fid,'    iStorey=0\n');
fprintf(fid,'    origin = joints[iStorey,iVertex,:]\n');
fprintf(fid,'    point2 = joints[iStorey,iVertex-1,:]\n');
fprintf(fid,'    name = ''Local_Datum_''+str(iVertex)\n');
fprintf(fid,'    LocalDatum_list.append(p_longerons.DatumCsysByThreePoints(origin=origin, point2=point2, name=name, \n');
fprintf(fid,'        coordSysType=CARTESIAN, point1=(0.0, 0.0, 0.0)))\n');
fprintf(fid,'    #\n');
fprintf(fid,'    # Then, create the longerons\n');
fprintf(fid,'    templist = [] # List that will contain the points used to make each longeron\n');
fprintf(fid,'    for iStorey in range(0,nStories+1,1):\n');
fprintf(fid,'        templist.append(joints[iStorey,iVertex,:])\n');
fprintf(fid,'        if iStorey != 0: # Save midpoints of bars\n');
fprintf(fid,'            long_midpoints.append( [(joints[iStorey-1,iVertex,:]+joints[iStorey,iVertex,:])/2 , ])\n');
fprintf(fid,'        # end if\n');
fprintf(fid,'    # end iStorey loop\n');
fprintf(fid,'    p_longerons.WirePolyLine(points=templist,\n');
fprintf(fid,'        mergeType=IMPRINT, meshable=ON)\n');
fprintf(fid,'    # Create set for each longeron (to assign local beam directions)\n');
fprintf(fid,'    for i in range(0,len(templist)): # loop over longerons edges\n');
fprintf(fid,'        if i == 0:\n');
fprintf(fid,'            select_edges = e_long.findAt([templist[0], ]) # Find the first edge\n');
fprintf(fid,'        else:\n');
fprintf(fid,'            # Now find remaining edges in longerons\n');
fprintf(fid,'            temp = e_long.findAt([templist[i], ])\n');
fprintf(fid,'            select_edges = select_edges + temp\n');
fprintf(fid,'        #end if\n');
fprintf(fid,'    #end i loop\n');
fprintf(fid,'    longeron_name = ''longeron-''+str(iVertex)+''_set''\n');
fprintf(fid,'    p_longerons.Set(edges=select_edges, name=longeron_name)\n');
fprintf(fid,'\n');
fprintf(fid,'#end for iVertex loop\n');
fprintf(fid,'\n');
fprintf(fid,'# Longerons set:\n');
fprintf(fid,'e_long = p_longerons.edges\n');
fprintf(fid,'select_edges = []\n');
fprintf(fid,'for i in range(0,len(long_midpoints)): # loop over longerons edges\n');
fprintf(fid,'    if i == 0:\n');
fprintf(fid,'        select_edges = e_long.findAt(long_midpoints[0]) # Find the first edge\n');
fprintf(fid,'    else:\n');
fprintf(fid,'        # Now find remaining edges in longerons\n');
fprintf(fid,'        temp = e_long.findAt(long_midpoints[i])\n');
fprintf(fid,'        select_edges = select_edges + temp\n');
fprintf(fid,'    #end if\n');
fprintf(fid,'\n');
fprintf(fid,'#end i loop\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons.Set(edges=select_edges, name=''all_longerons_set'')\n');
fprintf(fid,'all_longerons_set_edges = select_edges\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons.Surface(circumEdges=all_longerons_set_edges, name=''all_longerons_surface'')\n');
fprintf(fid,'\n');
% fprintf(fid,'# Create a transition zone where the longerons are thinner\n');
% fprintf(fid,'delta = 0.01*min(Longeron_b, Longeron_a)\n');
% fprintf(fid,'\n');
% fprintf(fid,'transition_longerons_set = e_long.getByBoundingBox(-MastRadius*1.2,-MastRadius*1.2,0.0-delta,MastRadius*1.2,MastRadius*1.2,MastHeight*transition_length_ratio+delta)\n');
% fprintf(fid,'p_longerons.Set(edges=transition_longerons_set, name=''transition_longerons_set'')\n');
% fprintf(fid,'\n');
% fprintf(fid,'remaining_longerons_set = e_long.getByBoundingBox(-MastRadius*1.2,-MastRadius*1.2,MastHeight*transition_length_ratio-delta,MastRadius*1.2,MastRadius*1.2,MastHeight+delta)\n');
% fprintf(fid,'p_longerons.Set(edges=remaining_longerons_set, name=''remaining_longerons_set'')\n');
% fprintf(fid,'\n');
% fprintf(fid,'# Create the battens:\n');
% fprintf(fid,'\n');
% fprintf(fid,'p_battens = mdb.models[''Model-1''].Part(name=''battens'', dimensionality=THREE_D, \n');
% fprintf(fid,'    type=DEFORMABLE_BODY)\n');
% fprintf(fid,'\n');
% fprintf(fid,'p_battens = mdb.models[''Model-1''].parts[''battens'']\n');
% fprintf(fid,'session.viewports[''Viewport: 1''].setValues(displayedObject=p_battens)\n');
% fprintf(fid,'\n');
% fprintf(fid,'d_battens, r_battens = p_battens.datums, p_battens.referencePoints\n');
% fprintf(fid,'\n');
% fprintf(fid,'\n');
% fprintf(fid,'if pinned_joints ==0:\n');
% fprintf(fid,'    storey_index = 1 # just an integer to define in which storey the battens start\n');
% fprintf(fid,'else:\n');
% fprintf(fid,'    storey_index = 0\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if\n');
% fprintf(fid,'batten_midpoints = []\n');
% fprintf(fid,'v_bat = p_battens.vertices\n');
% fprintf(fid,'\n');
% fprintf(fid,'if BattenFormat == 0: # Battens are a regular polygon linking longerons\n');
% fprintf(fid,'    for iStorey in range(1-storey_index,nStories+storey_index,1):\n');
% fprintf(fid,'        templist = [] # List that will contain each point used to make the long trusses\n');
% fprintf(fid,'        #for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'        for iVertex in range(VertexPolygon-1,-1,-1):\n');
% fprintf(fid,'            # Start batten at last joint:\n');
% fprintf(fid,'            templist.append(joints[iStorey,iVertex,:])\n');
% fprintf(fid,'            # compute batten vector and norm:\n');
% fprintf(fid,'            batten_vector = joints[iStorey,iVertex,:]-joints[iStorey,iVertex-1,:]\n');
% fprintf(fid,'            norm_bvec = numpy.linalg.norm(batten_vector)\n');
% fprintf(fid,'            # 1st point of batten used to attach a diagonal:\n');
% fprintf(fid,'            templist.append(joints[iStorey,iVertex-1,:]+batten_vector*Longeron_a/norm_bvec)\n');
% fprintf(fid,'            # save midpoint of first bar in this batten (for future selection)\n');
% fprintf(fid,'            batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*Longeron_a/(2.0*norm_bvec) ,  ])\n');
% fprintf(fid,'            # 2nd point to attach 2nd diagonal\n');
% fprintf(fid,'            templist.append(joints[iStorey,iVertex-1,:]+batten_vector*(1.0-Longeron_a/norm_bvec) ) \n');
% fprintf(fid,'            # save midpoing of 2nd bar of the batten (for future selection)\n');
% fprintf(fid,'            batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*0.5 ,  ])\n');
% fprintf(fid,'            # save midpoint of 3rd bar of the batten (for future selection)\n');
% fprintf(fid,'            batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*(1.0-Longeron_a/(2.0*norm_bvec)), ])\n');
% fprintf(fid,'            # End point of this batten\n');
% fprintf(fid,'            #if iVertex == 0: # find initial vertex\n');
% fprintf(fid,'            #    templist.append(v_bat.findAt(joints[iStorey,iVertex-1,:])) \n');
% fprintf(fid,'            #else: # Draw point\n');
% fprintf(fid,'            #    templist.append(joints[iStorey,iVertex-1,:]) \n');
% fprintf(fid,'            #if iVertex != 0: # Save midpoints of bars\n');
% fprintf(fid,'            #    batten_midpoints.append( [(joints[iStorey,iVertex-1,:]+joints[iStorey,iVertex,:])/2 , ])\n');
% fprintf(fid,'            # end if\n');
% fprintf(fid,'        # end iVertex loop\n');
% fprintf(fid,'        templist.append(joints[iStorey,-1,:]) # so that it connects to the initial vertex\n');
% fprintf(fid,'        #batten_midpoints.append( [(joints[iStorey,iVertex,:]+joints[iStorey,0,:])/2 , ]) # last bar in batten\n');
% fprintf(fid,'        p_battens.WirePolyLine(points=templist,\n');
% fprintf(fid,'            mergeType=IMPRINT, meshable=ON)\n');
% fprintf(fid,'    #end for iStorey loop\n');
% fprintf(fid,'else: # Battens are united in the middle, forming a star\n');
% fprintf(fid,'    for iStorey in range(1-storey_index,nStories+storey_index,1):\n');
% fprintf(fid,'        for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'            templist = [] # List that will contain each point used to make the long trusses\n');
% fprintf(fid,'            if iVertex == 0: # Create center point for the first bar in this batten:\n');
% fprintf(fid,'                templist.append((0.0,0.0,joints[iStorey,iVertex,2]))\n');
% fprintf(fid,'            else: # find center point previously created:\n');
% fprintf(fid,'                templist.append(v_bat.findAt((0.0,0.0,joints[iStorey,iVertex,2])))\n');
% fprintf(fid,'            #end if\n');
% fprintf(fid,'            templist.append(joints[iStorey,iVertex,:])\n');
% fprintf(fid,'            # Save midpoints of bars\n');
% fprintf(fid,'            batten_midpoints.append( [( (0.0,0.0,joints[iStorey,iVertex,2])+joints[iStorey,iVertex,:] )/2.0 , ])\n');
% fprintf(fid,'            # Create lines\n');
% fprintf(fid,'            p_battens.WirePolyLine(points=templist,\n');
% fprintf(fid,'                mergeType=IMPRINT, meshable=ON)\n');
% fprintf(fid,'        # end iVertex loop\n');
% fprintf(fid,'        #templist.append(joints[iStorey,0,:]) # so that it connects to the initial vertex\n');
% fprintf(fid,'        #batten_midpoints.append( [(joints[iStorey,iVertex,:]+joints[iStorey,0,:])/2 , ]) # last bar in batten\n');
% fprintf(fid,'        #p_longerons.WirePolyLine(points=templist,\n');
% fprintf(fid,'        #    mergeType=IMPRINT, meshable=ON)\n');
% fprintf(fid,'    #end for iStorey loop\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if BattenFormat\n');

% fprintf(fid,'batten_midpoints = []\n');
% fprintf(fid,'v_bat = p_battens.vertices\n');
% fprintf(fid,'for iStorey in range(1,nStories,1):\n');
% fprintf(fid,'    templist = [] # List that will contain each point used to make the long trusses\n');
% fprintf(fid,'    #for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'    for iVertex in range(VertexPolygon-1,-1,-1):\n');
% fprintf(fid,'        # Start batten at last joint:\n');
% fprintf(fid,'        templist.append(joints[iStorey,iVertex,:])\n');
% fprintf(fid,'        # compute batten vector and norm:\n');
% fprintf(fid,'        batten_vector = joints[iStorey,iVertex,:]-joints[iStorey,iVertex-1,:]\n');
% fprintf(fid,'        norm_bvec = numpy.linalg.norm(batten_vector)\n');
% fprintf(fid,'        # 1st point of batten used to attach a diagonal:\n');
% fprintf(fid,'        templist.append(joints[iStorey,iVertex-1,:]+batten_vector*Longeron_a/norm_bvec)\n');
% fprintf(fid,'        # save midpoint of first bar in this batten (for future selection)\n');
% fprintf(fid,'        batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*Longeron_a/(2.0*norm_bvec) ,  ])\n');
% fprintf(fid,'        # 2nd point to attach 2nd diagonal\n');
% fprintf(fid,'        templist.append(joints[iStorey,iVertex-1,:]+batten_vector*(1.0-Longeron_a/norm_bvec) ) \n');
% fprintf(fid,'        # save midpoing of 2nd bar of the batten (for future selection)\n');
% fprintf(fid,'        batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*0.5 ,  ])\n');
% fprintf(fid,'        # save midpoint of 3rd bar of the batten (for future selection)\n');
% fprintf(fid,'        batten_midpoints.append( [joints[iStorey,iVertex-1,:]+batten_vector*(1.0-Longeron_a/(2.0*norm_bvec)), ])\n');
% fprintf(fid,'        # End point of this batten\n');
% fprintf(fid,'        #if iVertex == 0: # find initial vertex\n');
% fprintf(fid,'        #    templist.append(v_bat.findAt(joints[iStorey,iVertex-1,:])) \n');
% fprintf(fid,'        #else: # Draw point\n');
% fprintf(fid,'        #    templist.append(joints[iStorey,iVertex-1,:]) \n');
% fprintf(fid,'        #if iVertex != 0: # Save midpoints of bars\n');
% fprintf(fid,'        #    batten_midpoints.append( [(joints[iStorey,iVertex-1,:]+joints[iStorey,iVertex,:])/2 , ])\n');
% fprintf(fid,'        # end if\n');
% fprintf(fid,'    # end iVertex loop\n');
% fprintf(fid,'    templist.append(joints[iStorey,-1,:]) # so that it connects to the initial vertex\n');
% fprintf(fid,'    #batten_midpoints.append( [(joints[iStorey,iVertex,:]+joints[iStorey,0,:])/2 , ]) # last bar in batten\n');
% fprintf(fid,'    p_battens.WirePolyLine(points=templist,\n');
% fprintf(fid,'        mergeType=IMPRINT, meshable=ON)\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end for iStorey loop\n');
fprintf(fid,'\n');
% fprintf(fid,'# battens set:\n');
% fprintf(fid,'e_bat = p_battens.edges\n');
% fprintf(fid,'select_edges = []\n');
% fprintf(fid,'for i in range(0,len(batten_midpoints)): # loop over remaining edges\n');
% fprintf(fid,'    if i == 0:\n');
% fprintf(fid,'        select_edges = e_bat.findAt(batten_midpoints[0]) # Find the first edge\n');
% fprintf(fid,'    else:\n');
% fprintf(fid,'        # Now find remaining edges in longerons\n');
% fprintf(fid,'        temp = e_bat.findAt(batten_midpoints[i])\n');
% fprintf(fid,'        select_edges = select_edges + temp\n');
% fprintf(fid,'    #end if\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end i loop\n');
% fprintf(fid,'\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    p_battens.Set(edges=select_edges, name=''all_battens_set'')\n');
% fprintf(fid,'    all_battens_set_edges = select_edges\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
% fprintf(fid,'\n');
fprintf(fid,'# Create a set with all the joints:\n');
fprintf(fid,'v_long = p_longerons.vertices\n');
% fprintf(fid,'v_bat = p_battens.vertices\n');
fprintf(fid,'select_vertices = []\n');
fprintf(fid,'select_top_vertices = []\n');
fprintf(fid,'select_bot_vertices = []\n');
fprintf(fid,'for iStorey in range(0,nStories+1,1):\n');
fprintf(fid,'    for iVertex in range(0,VertexPolygon,1):\n');
fprintf(fid,'        # Select all the joints in the longerons:\n');
fprintf(fid,'        current_joint = v_long.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the first vertex\n');
fprintf(fid,'        current_joint_name = ''joint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'        # Create a set for each joint:\n');
fprintf(fid,'        p_longerons.Set(vertices=current_joint, name=current_joint_name)\n');
fprintf(fid,'        #\n');
fprintf(fid,'        if iStorey == 0 and iVertex == 0:\n');
fprintf(fid,'            select_vertices = current_joint # Instantiate the first point in set\n');
fprintf(fid,'        else:\n');
fprintf(fid,'            select_vertices = select_vertices + current_joint # Instantiate the first point in set\n');
fprintf(fid,'        # endif iStorey == 0 and iVertex == 0\n');
fprintf(fid,'        #\n');
fprintf(fid,'        if iStorey == 0: # Also save the bottom nodes separately\n');
fprintf(fid,'            if iVertex == 0:\n');
fprintf(fid,'                # Start selecting the bottom joints for implementing the boundary conditions\n');
fprintf(fid,'                select_bot_vertices = current_joint\n');
fprintf(fid,'            else:\n');
fprintf(fid,'                select_bot_vertices = select_bot_vertices + current_joint\n');
fprintf(fid,'            # endif iStorey == 0:\n');
fprintf(fid,'        elif iStorey == nStories: # Also save the top nodes separately\n');
fprintf(fid,'            if iVertex == 0:\n');
fprintf(fid,'                # Start selecting the top joints for implementing the boundary conditions\n');
fprintf(fid,'                select_top_vertices = current_joint\n');
fprintf(fid,'            else: # remaining vertices:\n');
fprintf(fid,'                select_top_vertices = select_top_vertices + current_joint\n');
% fprintf(fid,'        else: # middle stories containing the battens (select the battens'' joints)\n');
% fprintf(fid,'            # Select points in battens:\n');
% fprintf(fid,'            current_joint = v_bat.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the first vertex\n');
% fprintf(fid,'            current_joint_name = ''joint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'            # Create a set for each joint:\n');
% fprintf(fid,'            p_battens.Set(vertices=current_joint, name=current_joint_name)\n');
fprintf(fid,'        #end if\n');
% fprintf(fid,'        # Select points in battens:\n');
% fprintf(fid,'        current_joint = v_bat.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the first vertex\n');
% fprintf(fid,'        current_joint_name = ''joint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'        # Create a set for each joint:\n');
% fprintf(fid,'        p_battens.Set(vertices=current_joint, name=current_joint_name)\n');
fprintf(fid,'    #end iVertex loop\n');
fprintf(fid,'\n');
fprintf(fid,'#end iStorey loop\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons.Set(vertices=select_vertices, name=''all_joints_set'')\n');
fprintf(fid,'p_longerons.Set(vertices=select_bot_vertices, name=''bot_joints_set'')\n');
fprintf(fid,'p_longerons.Set(vertices=select_top_vertices, name=''top_joints_set'')\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
% fprintf(fid,'# Create the diagonals:\n');
% fprintf(fid,'\n');
% fprintf(fid,'#p_diag1 = mdb.models[''Model-1''].Part(name=''diagonals1'', dimensionality=THREE_D, type=DEFORMABLE_BODY)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#p_diag1 = mdb.models[''Model-1''].parts[''diagonals1'']\n');
% fprintf(fid,'#session.viewports[''Viewport: 1''].setValues(displayedObject=p_diag1)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#d_diag1, r_diag1 = p_diag1.datums, p_diag1.referencePoints\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#p_diag2 = mdb.models[''Model-1''].Part(name=''diagonals2'', dimensionality=THREE_D, type=DEFORMABLE_BODY)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#p_diag2 = mdb.models[''Model-1''].parts[''diagonals2'']\n');
% fprintf(fid,'#session.viewports[''Viewport: 1''].setValues(displayedObject=p_diag2)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#d_diag2, r_diag2 = p_diag2.datums, p_diag2.referencePoints\n');
% fprintf(fid,'\n');
% fprintf(fid,'# Create diagonal trusses:\n');
% fprintf(fid,'if DiagonalRadius > 0.0: \n');
% fprintf(fid,'    diags_midpoints = []\n');
% fprintf(fid,'    v_bat = p_battens.vertices\n');
% fprintf(fid,'    for iStorey in range(1+3,nStories-1,1):\n');
% fprintf(fid,'        #templist = [] # List that will contain each point used to make the long trusses\n');
% fprintf(fid,'        #for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'        for iVertex in range(VertexPolygon-1,-1,-1):\n');
% fprintf(fid,'            # Start batten at last joint:\n');
% fprintf(fid,'            #templist.append(joints[iStorey,iVertex,:])\n');
% fprintf(fid,'            # compute batten vector and norm:\n');
% fprintf(fid,'            batten_vector = joints[iStorey,iVertex,:]-joints[iStorey,iVertex-1,:]\n');
% fprintf(fid,'            norm_bvec = numpy.linalg.norm(batten_vector)\n');
% fprintf(fid,'            # 1st point of batten used to attach a diagonal:\n');
% fprintf(fid,'            templist = []\n');
% fprintf(fid,'            pointA = joints[iStorey,iVertex-1,:]+batten_vector*Longeron_a/norm_bvec\n');
% fprintf(fid,'            templist.append(v_bat.findAt(pointA))\n');
% fprintf(fid,'            pointB = joints[iStorey+1,iVertex-1,:]+batten_vector*(1.0-Longeron_a/norm_bvec)\n');
% fprintf(fid,'            templist.append(v_bat.findAt(pointB))\n');
% fprintf(fid,'            diag_vector = pointB - pointA\n');
% fprintf(fid,'            norm_dvec = numpy.linalg.norm(diag_vector)\n');
% fprintf(fid,'            p_battens.WirePolyLine(points=templist, mergeType=IMPRINT, meshable=ON) # Draw first diagonal\n');
% fprintf(fid,'            diags_midpoints.append( [pointA + diag_vector*0.25,  ]) # save a point in the diagonal for future selection\n');
% fprintf(fid,'            diags_midpoints.append( [pointA + diag_vector*0.75,  ]) # save a point in the diagonal for future selection\n');
% fprintf(fid,'            #\n');
% fprintf(fid,'            # Now draw 2nd diagonal:\n');
% fprintf(fid,'            templist = []\n');
% fprintf(fid,'            pointA = joints[iStorey,iVertex-1,:]+batten_vector*(1.0-Longeron_a/norm_bvec)\n');
% fprintf(fid,'            templist.append(v_bat.findAt(pointA))\n');
% fprintf(fid,'            pointB = joints[iStorey+1,iVertex-1,:]+batten_vector*Longeron_a/norm_bvec\n');
% fprintf(fid,'            templist.append(v_bat.findAt(pointB))\n');
% fprintf(fid,'            diag_vector = pointB - pointA\n');
% fprintf(fid,'            norm_dvec = numpy.linalg.norm(diag_vector)\n');
% fprintf(fid,'            p_battens.WirePolyLine(points=templist, mergeType=IMPRINT, meshable=ON) # Draw first diagonal\n');
% fprintf(fid,'            diags_midpoints.append( [pointA + diag_vector*0.25,  ]) # save a point in the diagonal for future selection\n');
% fprintf(fid,'            diags_midpoints.append( [pointA + diag_vector*0.75,  ]) # save a point in the diagonal for future selection\n');
% fprintf(fid,'        # end iVertex loop\n');
% fprintf(fid,'    #end for iStorey loop\n');
% fprintf(fid,'    #\n');
% fprintf(fid,'    # Select all diagonals:\n');
% fprintf(fid,'    e_bat = p_battens.edges\n');
% fprintf(fid,'    select_edges = []\n');
% fprintf(fid,'    for i in range(0,len(diags_midpoints)): # loop over remaining edges\n');
% fprintf(fid,'        if i == 0:\n');
% fprintf(fid,'            select_edges = e_bat.findAt(diags_midpoints[0]) # Find the first edge\n');
% fprintf(fid,'        else:\n');
% fprintf(fid,'            # Now find remaining edges in longerons\n');
% fprintf(fid,'            temp = e_bat.findAt(diags_midpoints[i])\n');
% fprintf(fid,'            select_edges = select_edges + temp\n');
% fprintf(fid,'        #end if\n');
% fprintf(fid,'    #end i loop\n');
% fprintf(fid,'    p_battens.Set(edges=select_edges, name=''all_diagonals_set'')\n');
% fprintf(fid,'    all_diagonals_set_edges = select_edges\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if DiagonalRadius > 0.0\n');
% fprintf(fid,'\n');
fprintf(fid,'# Create materials:\n');
fprintf(fid,'mdb.models[''Model-1''].Material(name=''NiTi_alloy'')\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''NiTi_alloy''].Elastic(table=((83.0E3, 0.31), \n');
fprintf(fid,'    ))\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''NiTi_alloy''].Density(table=((1.0E-3, ), ))\n');
fprintf(fid,'\n');
fprintf(fid,'mdb.models[''Model-1''].Material(name=''PC'')\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''PC''].Elastic(table=((2134, 0.27), \n');
fprintf(fid,'    ))\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''PC''].Density(table=((1.19E-3, ), ))\n');
fprintf(fid,'\n');
fprintf(fid,'mdb.models[''Model-1''].Material(name=''PLA'')\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''PLA''].Elastic(table=((Emodulus, nu), \n');
% fprintf(fid,'mdb.models[''Model-1''].materials[''PLA''].Elastic(table=((2346.5, 0.27), \n');
% fprintf(fid,'mdb.models[''Model-1''].materials[''PLA''].Elastic(table=((1800, 0.27), \n');
fprintf(fid,'    ))\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''PLA''].Density(table=((1.24E-3, ), ))\n');
fprintf(fid,'\n');
fprintf(fid,'mdb.models[''Model-1''].Material(name=''CNT'')\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''CNT''].Elastic(table=((1000.0E3, 0.3), \n');
fprintf(fid,'    ))\n');
fprintf(fid,'mdb.models[''Model-1''].materials[''CNT''].Density(table=((1.0E-3, ), ))\n');
fprintf(fid,'\n');
fprintf(fid,'# Create beam profiles and beam sections:\n');
% fprintf(fid,'mdb.models[''Model-1''].CircularProfile(name=''LongeronsProfile'', r=Longeron_b)\n');
% fprintf(fid,'mdb.models[''Model-1''].CircularProfile(name=''LongeronsProfile2'', r=Longeron_b)\n');
% fprintf(fid,'#mdb.models[''Model-1''].RectangularProfile(name=''LongeronsProfile'', a=Longeron_a, b=Longeron_b*pi*3.0/8.0)\n');
% fprintf(fid,'mdb.models[''Model-1''].RectangularProfile(name=''LongeronsProfile'', a=Longeron_a, b=Longeron_b)\n');
% fprintf(fid,'mdb.models[''Model-1''].RectangularProfile(name=''LongeronsProfile2'', a=Longeron_a/1.0, b=Longeron_b/1.0)\n');
fprintf(fid,'mdb.models[''Model-1''].GeneralizedProfile(name=''LongeronsProfile'', area=Longeron_CS, i11=Ix, i12=0.0, i22=Iy, j=J, gammaO=0.0, gammaW=0.0)\n');
% fprintf(fid,'mdb.models[''Model-1''].GeneralizedProfile(name=''LongeronsProfile2'', area=Longeron_CS, i11=Ix, i12=0.0, i22=Iy, j=J, gammaO=0.0, gammaW=0.0)\n');
% fprintf(fid,'#mdb.models[''Model-1''].BoxProfile(name=''LongeronsProfile'', a=Longeron_a, b=Longeron_b,uniformThickness=ON, t1=Longeron_b/4.0)\n');
% fprintf(fid,'#mdb.models[''Model-1''].PipeProfile(name=''LongeronsProfile'', r=Longeron_b, t=1.0)\n');
fprintf(fid,'\n');
% fprintf(fid,'mdb.models[''Model-1''].BeamSection(consistentMassMatrix=False, integration=\n');
% fprintf(fid,'    DURING_ANALYSIS, material=Longeron_Material, name=''LongeronsSection'', poissonRatio=0.31, \n');
% fprintf(fid,'    profile=''LongeronsProfile'', temperatureVar=LINEAR)\n');
% fprintf(fid,'\n');
% fprintf(fid,'mdb.models[''Model-1''].BeamSection(consistentMassMatrix=False, integration=\n');
% fprintf(fid,'    DURING_ANALYSIS, material=Longeron_Material, name=''LongeronsSection2'', poissonRatio=0.31, \n');
% fprintf(fid,'    profile=''LongeronsProfile2'', temperatureVar=LINEAR)\n');
% fprintf(fid,'#\n');
fprintf(fid,'mdb.models[''Model-1''].BeamSection(name=''LongeronsSection'', integration=\n');
fprintf(fid,'      BEFORE_ANALYSIS, poissonRatio=0.31, beamShape=CONSTANT, \n'); % MIGHT REQUIRE CHANGE OF NU
fprintf(fid,'         profile=''LongeronsProfile'', density=0.00124, thermalExpansion=OFF, \n'); 
fprintf(fid,'         temperatureDependency=OFF, dependencies=0, table=((Emodulus, Gmodulus), ), \n'); 
fprintf(fid,'         alphaDamping=0.0, betaDamping=0.0, compositeDamping=0.0, centroid=(0.0, \n'); 
fprintf(fid,'         0.0), shearCenter=(0.0, 0.0), consistentMassMatrix=False)\n');
fprintf(fid,'\n');
% fprintf(fid,'mdb.models[''Model-1''].BeamSection(name=''LongeronsSection2'', integration=\n');
% fprintf(fid,'      BEFORE_ANALYSIS, poissonRatio=0.31, beamShape=CONSTANT, \n');% MIGHT REQUIRE CHANGE OF NU
% fprintf(fid,'         profile=''LongeronsProfile2'', density=0.00124, thermalExpansion=OFF, \n'); 
% fprintf(fid,'         temperatureDependency=OFF, dependencies=0, table=((Emodulus, Gmodulus), ), \n'); 
% fprintf(fid,'         alphaDamping=0.0, betaDamping=0.0, compositeDamping=0.0, centroid=(0.0, \n'); 
% fprintf(fid,'         0.0), shearCenter=(0.0, 0.0), consistentMassMatrix=False)\n');
% fprintf(fid,'\n');
% fprintf(fid,'mdb.models[''Model-1''].CircularProfile(name=''BattensProfile'', r=BattenRadius)\n');
% fprintf(fid,'#mdb.models[''Model-1''].PipeProfile(name=''BattensProfile'', r=BattenRadius, t=1.0)\n');
% fprintf(fid,'\n');
% fprintf(fid,'mdb.models[''Model-1''].BeamSection(consistentMassMatrix=False, integration=\n');
% fprintf(fid,'    DURING_ANALYSIS, material=Batten_Material, name=''BattensSection'', poissonRatio=0.31, \n');
% fprintf(fid,'    profile=''BattensProfile'', temperatureVar=LINEAR)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'\n');
% fprintf(fid,'if DiagonalRadius > 0.0: \n');
% fprintf(fid,'    mdb.models[''Model-1''].CircularProfile(name=''DiagonalsProfile'', r=DiagonalRadius)\n');
% fprintf(fid,'    #mdb.models[''Model-1''].PipeProfile(name=''DiagonalsProfile'', r=DiagonalRadius, t=1.0)\n');
% fprintf(fid,'    \n');
% fprintf(fid,'    mdb.models[''Model-1''].BeamSection(consistentMassMatrix=False, integration=\n');
% fprintf(fid,'        DURING_ANALYSIS, material=Diagonal_Material, name=''DiagonalsSection'', poissonRatio=0.31, \n');
% fprintf(fid,'        profile=''DiagonalsProfile'', temperatureVar=LINEAR)\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if DiagonalRadius > 0.0:\n');
% fprintf(fid,'\n');
fprintf(fid,'# Assign respective sections:\n');
fprintf(fid,'p_longerons.SectionAssignment(offset=0.0, \n');
fprintf(fid,'    offsetField='''', offsetType=MIDDLE_SURFACE, region=\n');
fprintf(fid,'    p_longerons.sets[''all_longerons_set''], \n');
fprintf(fid,'    sectionName=''LongeronsSection'', thicknessAssignment=FROM_SECTION)\n');
fprintf(fid,'\n');
% fprintf(fid,'p_longerons.SectionAssignment(offset=0.0, \n');
% fprintf(fid,'    offsetField='''', offsetType=MIDDLE_SURFACE, region=\n');
% fprintf(fid,'    p_longerons.sets[''transition_longerons_set''], \n');
% fprintf(fid,'    sectionName=''LongeronsSection2'', thicknessAssignment=FROM_SECTION)\n');
% fprintf(fid,'\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    p_battens.SectionAssignment(offset=0.0, \n');
% fprintf(fid,'        offsetField='''', offsetType=MIDDLE_SURFACE, region=\n');
% fprintf(fid,'        p_battens.sets[''all_battens_set''], \n');
% fprintf(fid,'        sectionName=''BattensSection'', thicknessAssignment=FROM_SECTION)\n');
% fprintf(fid,'    #\n');
% fprintf(fid,'    if DiagonalRadius > 0.0: \n');
% fprintf(fid,'        p_battens.SectionAssignment(offset=0.0, \n');
% fprintf(fid,'            offsetField='''', offsetType=MIDDLE_SURFACE, region=\n');
% fprintf(fid,'            p_battens.sets[''all_diagonals_set''], \n');
% fprintf(fid,'            sectionName=''DiagonalsSection'', thicknessAssignment=FROM_SECTION)\n');
% fprintf(fid,'    #end if DiagonalRadius > 0.0\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
% fprintf(fid,'\n');
% fprintf(fid,'#\n');
fprintf(fid,'# Assing beam orientation:\n');
fprintf(fid,'for iVertex in range(0,VertexPolygon,1):\n');
fprintf(fid,'    iStorey=0\n');
fprintf(fid,'    dir_vec_n1 = joints[iStorey,iVertex,:]-(0.,0.,0.) # Vector n1 perpendicular to the longeron tangent\n');
fprintf(fid,'    longeron_name = ''longeron-''+str(iVertex)+''_set''\n');
fprintf(fid,'    region=p_longerons.sets[longeron_name]\n');
fprintf(fid,'    p_longerons.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=dir_vec_n1)\n');
fprintf(fid,'\n');
fprintf(fid,'#end for iVertex\n');
fprintf(fid,'#\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    region=p_battens.sets[''all_battens_set'']\n');
% fprintf(fid,'    p_battens.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, \n');
% fprintf(fid,'        1.0))\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
% fprintf(fid,'#\n');
% fprintf(fid,'\n');
% fprintf(fid,'if DiagonalRadius > 0.0: \n');
% fprintf(fid,'    region=p_battens.sets[''all_diagonals_set'']\n');
% fprintf(fid,'    p_battens.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, \n');
% fprintf(fid,'        1.0))\n');
fprintf(fid,'\n');
% fprintf(fid,'#end if DiagonalRadius > 0.0\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#\n');
fprintf(fid,'delta = Mesh_size/100.0\n');
fprintf(fid,'########################################################################\n');
fprintf(fid,'#Mesh the structure\n');
fprintf(fid,'\n');
fprintf(fid,'#refPlane = p_longerons.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=L/2)\n');
fprintf(fid,'#d = p.datums\n');
fprintf(fid,'#All_faces = facesLeafs+facesDoubleThickBoom\n');
fprintf(fid,'#p.PartitionFaceByDatumPlane(datumPlane=d[refPlane.id], faces=All_faces)\n');
fprintf(fid,'##\n');
fprintf(fid,'#session.viewports[''Viewport: 1''].partDisplay.setValues(sectionAssignments=OFF\n');
fprintf(fid,'#    engineeringFeatures=OFF, mesh=ON)\n');
fprintf(fid,'#session.viewports[''Viewport: 1''].partDisplay.meshOptions.setValues(\n');
fprintf(fid,'#    meshTechnique=ON)\n');
fprintf(fid,'#p = mdb.models[''Model-1''].parts[''reducedCF_TRAC_boom'']\n');
fprintf(fid,'\n');
fprintf(fid,'p_longerons.seedPart(size=Mesh_size, deviationFactor=0.04, minSizeFactor=0.001,\n');
fprintf(fid,'    constraint=FINER)\n');
% fprintf(fid,'p_longerons.seedPart(size=Mesh_size, deviationFactor=0.04, minSizeFactor=0.04)\n');
% fprintf(fid,'p_longerons.seedEdgeBySize(edges=all_longerons_set_edges, size=Mesh_size, deviationFactor=0.4,\n');
% fprintf(fid,'    constraint=FINER)\n');
fprintf(fid,'p_longerons.seedEdgeBySize(edges=all_longerons_set_edges, size=Mesh_size, deviationFactor=0.04,\n');
fprintf(fid,'    constraint=FINER)\n');
fprintf(fid,'elemType_longerons = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD) # Element type\n');
fprintf(fid,'p_longerons.setElementType(regions=(all_longerons_set_edges, ), elemTypes=(elemType_longerons, ))\n');
fprintf(fid,'p_longerons.generateMesh()\n');
fprintf(fid,'\n');
% fprintf(fid,'# Mesh battens and diagonals:\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    p_battens.seedPart(size=Mesh_size, deviationFactor=0.04, minSizeFactor=0.04)\n');
% fprintf(fid,'    # Just battens:\n');
% fprintf(fid,'    p_battens.seedEdgeBySize(edges=all_battens_set_edges, size=Mesh_size, deviationFactor=0.4,\n');
% fprintf(fid,'        constraint=FINER)\n');
% fprintf(fid,'    elemType_battens = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD) # Element type\n');
% fprintf(fid,'    p_battens.setElementType(regions=(all_battens_set_edges, ), elemTypes=(elemType_battens, ))\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
% fprintf(fid,'\n');
% fprintf(fid,'# Just diagonals:\n');
% fprintf(fid,'if DiagonalRadius > 0.0:\n');
% fprintf(fid,'    p_battens.seedEdgeBySize(edges=all_battens_set_edges, size=Mesh_size, deviationFactor=0.4,\n');
% fprintf(fid,'        constraint=FINER)\n');
% fprintf(fid,'    elemType_diagonals = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD) # Element type\n');
% fprintf(fid,'    p_battens.setElementType(regions=(all_diagonals_set_edges, ), elemTypes=(elemType_battens, ))\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if DiagonalRadius > 0.0\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    p_battens.generateMesh()\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
fprintf(fid,'#######################################################################\n');
fprintf(fid,'\n');
fprintf(fid,'# Make Analytical surfaces for contact purposes\n');
fprintf(fid,'s1 = mdb.models[''Model-1''].ConstrainedSketch(name=''__profile__'', \n');
fprintf(fid,'    sheetSize=MastRadius*3.0)\n');
fprintf(fid,'g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints\n');
fprintf(fid,'s1.setPrimaryObject(option=STANDALONE)\n');
fprintf(fid,'s1.Line(point1=(0.0, -MastRadius*1.1), point2=(0.0, MastRadius*1.1))\n');
fprintf(fid,'s1.VerticalConstraint(entity=g[2], addUndoState=False)\n');
fprintf(fid,'p_surf = mdb.models[''Model-1''].Part(name=''AnalyticSurf'', dimensionality=THREE_D, \n');
fprintf(fid,'    type=ANALYTIC_RIGID_SURFACE)\n');
fprintf(fid,'p_surf = mdb.models[''Model-1''].parts[''AnalyticSurf'']\n');
fprintf(fid,'p_surf.AnalyticRigidSurfExtrude(sketch=s1, depth=MastRadius*2.2)\n');
fprintf(fid,'s1.unsetPrimaryObject()\n');
fprintf(fid,'\n');
fprintf(fid,'rigid_face = p_surf.faces\n');
fprintf(fid,'#surf_select = f.findAt((0.0,MastRadius*1.05,0.0))\n');
fprintf(fid,'#surf_select = f[0]\n');
fprintf(fid,'p_surf.Surface(side1Faces=rigid_face, name=''rigid_support'')\n');
fprintf(fid,'#p_surf.Set(faces=surf_select, name=''support_surface_set'')\n');
fprintf(fid,'#p_surf.sets[''all_diagonals_set'']\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
fprintf(fid,'# Make assembly:\n');
fprintf(fid,'a = mdb.models[''Model-1''].rootAssembly\n');
fprintf(fid,'a.DatumCsysByDefault(CARTESIAN)\n');
fprintf(fid,'# Create reference points to assign boundary conditions\n');
fprintf(fid,'RP_ZmYmXm = a.ReferencePoint(point=(0.0, 0.0, -1.1*MastRadius))\n');
fprintf(fid,'refpoint_ZmYmXm = (a.referencePoints[RP_ZmYmXm.id],)\n');
fprintf(fid,'a.Set(referencePoints=refpoint_ZmYmXm, name=''RP_ZmYmXm'')\n');
fprintf(fid,'#\n');
fprintf(fid,'RP_ZpYmXm = a.ReferencePoint(point=(0.0, 0.0, MastHeight+1.1*MastRadius))\n');
fprintf(fid,'refpoint_ZpYmXm = (a.referencePoints[RP_ZpYmXm.id],)\n');
fprintf(fid,'a.Set(referencePoints=refpoint_ZpYmXm, name=''RP_ZpYmXm'')\n');
fprintf(fid,'#\n');
fprintf(fid,'# Create longerons\n');
fprintf(fid,'a_long = a.Instance(name=''longerons-1-1'', part=p_longerons, dependent=ON)\n');
% fprintf(fid,'if (nStories > 1) or (pinned_joints == 0):\n');
% fprintf(fid,'    # Create battens\n');
% fprintf(fid,'    a_bat = a.Instance(name=''battens-1-1'', part=p_battens, dependent=ON)\n');
% fprintf(fid,'\n');
% fprintf(fid,'#end if nStories\n');
fprintf(fid,'# Create bottom surface\n');
fprintf(fid,'a_surf_bot = a.Instance(name=''AnalyticSurf-1-1'', part=p_surf, dependent=ON)\n');
fprintf(fid,'# Now rotate the plane to have the proper direction\n');
fprintf(fid,'a.rotate(instanceList=(''AnalyticSurf-1-1'', ), axisPoint=(0.0, 0.0, 0.0), \n');
fprintf(fid,'    axisDirection=(0.0, 1.0, 0.0), angle=90.0)\n');
fprintf(fid,'#\n');
fprintf(fid,'# Create set with surface\n');
% fprintf(fid,'#f = a_surf_bot.faces\n');
% fprintf(fid,'#surf_select = f.findAt((0.0,MastRadius*1.05,0.0))\n');
% fprintf(fid,'#select_bot_surf=a.instances[''AnalyticSurf-1-1''].surfaces[''rigid_support'']\n');
fprintf(fid,'select_bot_surf=a_surf_bot.surfaces[''rigid_support'']\n');
fprintf(fid,'# Perhaps we need to define a set instead of a face\n');
fprintf(fid,'#AnalyticSurf_surface=a_surf_bot.Surface(side1Faces=select_bot_surf, name=''support_surf_bot-1'')\n');
fprintf(fid,'mdb.models[''Model-1''].RigidBody(name=''Constraint-RigidBody_surf_bot-1'', refPointRegion=refpoint_ZmYmXm, \n');
fprintf(fid,'    surfaceRegion=select_bot_surf)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#mdb.models[''Model-1''].ContactProperty(''IntProp-1'')\n');
% fprintf(fid,'#mdb.models[''Model-1''].interactionProperties[''IntProp-1''].TangentialBehavior(\n');
% fprintf(fid,'#    formulation=FRICTIONLESS)\n');
% fprintf(fid,'#mdb.models[''Model-1''].interactionProperties[''IntProp-1''].NormalBehavior(\n');
% fprintf(fid,'#    pressureOverclosure=HARD, allowSeparation=ON, \n');
% fprintf(fid,'#    constraintEnforcementMethod=DEFAULT)\n');
% fprintf(fid,'##: The interaction property "IntProp-1" has been created.\n');
% fprintf(fid,'#a = mdb.models[''Model-1''].rootAssembly\n');
% fprintf(fid,'#slave_region=a.instances[''longerons-1-1''].sets[''all_longerons_set'']\n');
% fprintf(fid,'#mdb.models[''Model-1''].SurfaceToSurfaceContactStd(name=''Int-1'', \n');
% fprintf(fid,'#    createStepName=''Initial'', master=select_bot_surf, slave=slave_region, sliding=FINITE, \n');
% fprintf(fid,'#    thickness=OFF, interactionProperty=''IntProp-1'', surfaceSmoothing=NONE, \n');
% fprintf(fid,'#    adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, \n');
% fprintf(fid,'#    clearanceRegion=None)\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#\n');
% fprintf(fid,'#for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'#    # Bottom base:\n');
% fprintf(fid,'#    iStorey = 0\n');
% fprintf(fid,'#    RP_temp = a.ReferencePoint(point=tuple(joints[iStorey,iVertex,:]))\n');
% fprintf(fid,'#    refpoint_temp = (a.referencePoints[RP_temp.id],)\n');
% fprintf(fid,'#    RP_temp_name = ''RP_joint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'#    a.Set(referencePoints=refpoint_temp, name=RP_temp_name)\n');
% fprintf(fid,'#    \n');
% fprintf(fid,'#\n');
% fprintf(fid,'#\n');
fprintf(fid,'for iVertex in range(0,VertexPolygon,1):\n');
% fprintf(fid,'#    iStorey=0\n');
% fprintf(fid,'#    origin = joints[iStorey,iVertex,:]\n');
% fprintf(fid,'#    point2 = joints[iStorey,iVertex-1,:]\n');
% fprintf(fid,'#    name = ''Local_Datum_''+str(iVertex)\n');
% fprintf(fid,'#    LocalDatum_list.append(a.DatumCsysByThreePoints(origin=origin, point2=point2, name=name, \n');
% fprintf(fid,'#        coordSysType=CARTESIAN, point1=(0.0, 0.0, 0.0)))\n');
fprintf(fid,'    #\n');
fprintf(fid,'    # Select appropriate coordinate system:\n');
fprintf(fid,'    DatumID = LocalDatum_list[iVertex].id\n');
fprintf(fid,'    datum = a_long.datums[DatumID]\n');
fprintf(fid,'    for iStorey in range(0,nStories+1,1):\n');
fprintf(fid,'        # Current joint:\n');
fprintf(fid,'        current_joint_name = ''joint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'        # Define COUPLING constraints for all the joints:\n');
fprintf(fid,'        if iStorey == 0: # Bottom base:\n');
fprintf(fid,'            #\n');
fprintf(fid,'            master_region=a.sets[''RP_ZmYmXm''] # Note that the master is the Reference Point\n');
fprintf(fid,'            #\n');
fprintf(fid,'            slave_region=a_long.sets[current_joint_name]\n');
fprintf(fid,'            # Make constraint for this joint:\n');
fprintf(fid,'            Constraint_name = ''RP_ZmYmXm_PinConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'            mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
fprintf(fid,'                surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
fprintf(fid,'                localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)\n');
fprintf(fid,'            #\n');
fprintf(fid,'            #Constraint_name = ''RP_ZmYmXm_FixedConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'            #mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
fprintf(fid,'            #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
fprintf(fid,'            #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)\n');
fprintf(fid,'            # Make constraint for this joint:\n');
% fprintf(fid,'            if pinned_joints == 1:# Battens are pinned to longerons\n');
% fprintf(fid,'                Constraint_name = ''PinConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            elif pinned_joints == 0:# Battens are welded to longerons\n');
% fprintf(fid,'            if pinned_joints == 0:# Battens are welded to longerons\n');
% fprintf(fid,'                master_region=a_long.sets[current_joint_name]\n');
% fprintf(fid,'                #\n');
% fprintf(fid,'                slave_region=a_bat.sets[current_joint_name]\n');
% fprintf(fid,'                Constraint_name = ''FixedConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            #endif pinned\n');
fprintf(fid,'        elif iStorey == nStories: # Top base:\n');
fprintf(fid,'            #\n');
fprintf(fid,'            master_region=a.sets[''RP_ZpYmXm''] # Note that the master is the Reference Point\n');
fprintf(fid,'            #\n');
fprintf(fid,'            slave_region=a_long.sets[current_joint_name]\n');
fprintf(fid,'            # Make constraint for this joint:\n');
fprintf(fid,'            Constraint_name = ''RP_ZpYmXm_PinConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'            mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
fprintf(fid,'                surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
fprintf(fid,'                localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)\n');
fprintf(fid,'            #\n');
fprintf(fid,'            #Constraint_name = ''RP_ZpYmXm_FixedConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
fprintf(fid,'            #mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
fprintf(fid,'            #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
fprintf(fid,'            #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)\n');
fprintf(fid,'            # Make constraint for this joint:\n');
% fprintf(fid,'            if pinned_joints == 1:# Battens are pinned to longerons\n');
% fprintf(fid,'                Constraint_name = ''PinConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            elif pinned_joints == 0:# Battens are welded to longerons\n');
% fprintf(fid,'            if pinned_joints == 0:# Battens are welded to longerons\n');
% fprintf(fid,'                master_region=a_long.sets[current_joint_name]\n');
% fprintf(fid,'                #\n');
% fprintf(fid,'                slave_region=a_bat.sets[current_joint_name]\n');
% fprintf(fid,'                Constraint_name = ''FixedConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            #endif pinned\n');
fprintf(fid,'        else: # Middle stories:\n');
% fprintf(fid,'            #v_long = a_long.vertices # vertices from longerons\n');
% fprintf(fid,'            #master_point = v_long.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the longeron joint\n');
% fprintf(fid,'            #master_name = ''LongPinPoint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'            #master_region=a.Set(vertices=master_point, name=master_name) # Longeron set to become master\n');
% fprintf(fid,'            #v_bat = a_bat.vertices # vertices from battens\n');
% fprintf(fid,'            #slave_point = v_bat.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the batten joint\n');
% fprintf(fid,'            #slave_name = ''BatPinPoint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'            #slave_region=a.Set(vertices=slave_point, name=slave_name) # Batten set to become master\n');
fprintf(fid,'            master_region=a_long.sets[current_joint_name]\n');
fprintf(fid,'            #\n');
fprintf(fid,'            slave_region=a_bat.sets[current_joint_name]\n');
fprintf(fid,'            # Make constraint for this joint:\n');
% fprintf(fid,'            if pinned_joints == 1:# Battens are pinned to longerons\n');
% fprintf(fid,'                Constraint_name = ''PinConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            elif pinned_joints == 0:# Battens are welded to longerons\n');
% fprintf(fid,'            #\n');
% fprintf(fid,'                Constraint_name = ''FixedConstraint-''+str(iStorey)+''-''+str(iVertex)\n');
% fprintf(fid,'                mdb.models[''Model-1''].Coupling(name=Constraint_name, controlPoint=master_region, \n');
% fprintf(fid,'                    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, \n');
% fprintf(fid,'                    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)\n');
% fprintf(fid,'            #endif pinned\n');
fprintf(fid,'        #endif iStorey\n');
fprintf(fid,'        #\n');
fprintf(fid,'    #end for iStorey\n');
fprintf(fid,'\n');
fprintf(fid,'#end for iVertex\n');
fprintf(fid,'\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
fprintf(fid,'\n');
fprintf(fid,'# Create hinges:\n');
fprintf(fid,'#select_joints=a.instances[''deployable_mast-1''].sets[''all_joints_set'']\n');
fprintf(fid,'#select_RefPoint=a.sets[''RP_joints'']\n');
fprintf(fid,'#mdb.models[''Model-1''].RigidBody(name=''JointsContraint'', refPointRegion=select_RefPoint, \n');
fprintf(fid,'#    pinRegion=select_joints)\n');
fprintf(fid,'\n');
fprintf(fid,'#\n');
fprintf(fid,'# Export mesh to .inp file\n');
fprintf(fid,'#\n');
fprintf(fid,'mdb.Job(name=''include_mesh_DoE%i'', model=''Model-1'', type=ANALYSIS, explicitPrecision=SINGLE,\n',jDoE);
fprintf(fid,'    nodalOutputPrecision=SINGLE, description='''',\n');
fprintf(fid,'    parallelizationMethodExplicit=DOMAIN, multiprocessingMode=DEFAULT,\n');
fprintf(fid,'    numDomains=1, userSubroutine='''', numCpus=1, memory=90,\n');
fprintf(fid,'    memoryUnits=PERCENTAGE, scratch='''', echoPrint=OFF, modelPrint=OFF,\n');
fprintf(fid,'    contactPrint=OFF, historyPrint=OFF)\n');
fprintf(fid,'import os\n');
fprintf(fid,'mdb.jobs[''include_mesh_DoE%i''].writeInput(consistencyChecking=OFF)\n',jDoE);
fprintf(fid,'# End of python script\n');
fprintf(fid,'\n');



% fprintf(fid,'# End of python script\n');
%
%
fclose(fid);
%
% disp(' ');
% disp('Creation of Python file COMPLETED');
% disp('Elapsed Time [min]: ');
% disp(toc/60);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute Python Script in ABAQUS CAE                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
disp(' ');
disp(['Started generation of mesh files for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]); disp(' ');
unix(strcat(UserSettings.abaqus_path,' cae noGUI=',jDoE_dir,'/script_DoE',num2str(jDoE),'_meshing.py'));
disp(' ');
disp(['Generation of mesh COMPLETED for DoE',num2str(jDoE),' Imperfection',num2str(kImperfection)]);
% disp('Elapsed Time [min]: ');
% disp(toc/60);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of Function                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A few notes to read the buckling values:

% import re
% match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
% eigenValue = [float(x) for x in re.findall(match_number, RVEframe.description)]
% eigenValue[1] is the eigenValue for frame = eigenValue[0]

%
end
