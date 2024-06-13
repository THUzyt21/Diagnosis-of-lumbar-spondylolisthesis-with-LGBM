import math  
import numpy as np
import pandas as pd
import copy 
def clockwiseangle_and_distance(origin,point):
    refvec = [1, 0]
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    # if angle < 0:
    #     return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector
# 保证点是顺时针排列，第一个点位于左上
def Clockwise(pts):
    pts = copy.deepcopy(pts)
    pts = np.array(pts).reshape((-1, 2))
    # 根据x进行排序
    pts = pts[pts[:, 0].argsort()]
    left_two_points = pts[:2].copy()
    right_two_points = pts[2:].copy()
    # 根据y进行排序
    left_two_points = left_two_points[left_two_points[:, 1].argsort()]
    origin = left_two_points[0]
    rest = np.concatenate([right_two_points, left_two_points[[1]]], axis=0)
    angle = []
    for box in rest:
        angle.append(clockwiseangle_and_distance(origin, box)[0])
    angle = np.array(angle)
    order = np.array(angle).argsort()[::-1]
    rest = rest[order]
    rect = np.vstack((origin, rest))
    # return the ordered coordinates
    return np.reshape(rect, -1).tolist()
def re_index(x): #对同一张图片的框进行排序
    tmp = x[:]
    test = sorted(tmp, key = lambda x:x[1])
    for i in range(len(x)):
        x[i] = test[i]
    return x

def inline_points_transfer(points):
    if len(points) == 8:
        a=[points[0], points[2], points[4], points[6]]
        b=[points[1], points[3], points[5], points[7]]
    else:
        a=[points[0], points[2], points[4]]
        b=[points[1], points[3], points[5]]
    return a,b
#points之间的转换
def point_inner_transfer(data):
    res=[]
    for i in range(len(data)):
        tmp={}
        tmp['name'] = 'polygon'
        r=inline_points_transfer(data[i])
        tmp['all_points_x'] = r[0]
        tmp['all_points_y'] = r[1]
        res.append(tmp)
    return res

#修正：1.根据人的朝向，把朝向为左的数据进行翻折
def mend_one_list(a,direction, j):#翻折函数
    #先进行找左上角点的修正
    if type(a) == str: return a
    #for i in range(len(a)//2):
    #    a[i],a[len(a)-i-1] = a[len(a)-i-1], a[i]
    #1 2 3 4 ->  2 1 4 3
    if(len(a) == 4):
        a[0],a[1] = a[1],a[0]
        a[2],a[3] = a[3],a[2]
    #所有的x坐标用2160相减，做一个翻折
    if j:
        for i in range(len(a)):
            a[i] = 2160 - a[i]
    return a
#修正box6
def mend_box6(l5_3, box6_ori, direct):
    box6 = copy.deepcopy(box6_ori)
    points = list(zip(box6["all_points_x"], box6["all_points_y"]))
    '''
    dists = []
    for p in points:
        dists.append( (p[0] - l5_3[0])**2 + (p[1] - l5_3[1])**2 )
    point1 = points[ dists.index(min(dists))]
    points.remove(point1)
    point2 = max(points, key = lambda x: x[1])
    points.remove(point2)
    point3 = points[0]
   # if direct == -1:
   #     point2,point3 = point3, point2
    if point1[0] > point2[0]:
        point1,point2 = point2,point1
    pois = list(zip(point1,point2,point3))
    box6["all_points_x"] = list(pois[0])
    box6["all_points_y"] = list(pois[1])
    '''
    points.sort(key = lambda x:x[0])
    point1 = points[1]
    point2 = points[2]
    point3 = points[0]
    pois = list(zip(point1,point2,point3))
    box6['all_points_x'] = list(pois[0])
    box6['all_points_y'] = list(pois[1])
    return box6

def getLinearEquation(p1x, p1y, p2x, p2y): #输入3点->输出直线方程的A,B,C
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]

def get_angle_and_perpdistanceRatio(a,b,c,d):#a,b,c,d分别代表 2 3 4 5这4个点,输出滑脱系数和垂足坐标
    point2,point3,point4,point5 = a,b,c,d
    vec_up = np.array([point2[0]-point3[0], point2[1]-point3[1]])
    vec_down = np.array([point5[0]-point4[0], point5[1] - point4[1]])
    #得到45这条直线的一般式方程 Ax+By+C = 0
    [A,B,C] = getLinearEquation(point4[0], point4[1], point5[0], point5[1])
    #计算垂足的坐标
    x0,y0 = point2[0],point2[1]
    px = (B*B*x0 - A*B*y0 - A*C)/(A*A+B*B)
    py = (A*A*y0 - A*B*x0 - B*C)/(A*A+B*B)
    distance = np.linalg.norm([px-point4[0], py-point4[1]], 2) * (px-point4[0])/abs(px-point4[0])
    length45 = np.linalg.norm(vec_down,2) 
    return 0,distance/length45, px,py 
def distance(A,B):
    return (A[0]-B[0])**2 + (A[1]-B[1])**2
def get_slope_and_length(A,B):
    if A[0]==B[0]: return (90, round(math.sqrt(distance(A,B)),4))
    slope = round((A[1] - B[1])/(A[0] - B[0]), 4)
    return (round(slope,2), round(math.sqrt(distance(A,B)),4))

def get_PSD(point1, point2,point4):
    # 计算两条线的斜率
    if point2[0] != point1[0]:
        slope1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        slope1 = float('inf')  # 如果垂直线,斜率设为无穷大
    if point4[0] != point2[0]:
        slope2 = (point4[1] - point2[1]) / (point4[0] - point2[0])
    else:
        slope2 = float('inf')
    # 计算两条线与水平方向的夹角
    angle1 = math.atan(slope1)
    angle2 = math.atan(slope2)
    # 将角度转换为0到90度范围内
    if angle1 < 0:
        angle1 += math.pi / 2
    else:
        angle1 = math.pi / 2 - angle1
    if angle2 < 0:
        angle2 += math.pi / 2
    else:
        angle2 = math.pi / 2 - angle2
    # 返回两个角度差的绝对值
    return math.degrees(abs(angle1-angle2))
def get_info(idx,pic_num,picture):
    #找到两块脊柱的关键点
    #上脊柱spine1 下脊柱spine2
    if idx<2: return 0
    spine1 = picture[idx-2]
    spine2 = picture[idx-1]
    if(len(spine1['all_points_x']) <= 3): return 0
    #编号为point 1 2 3 4 5 6
    point_dict = {}
    '''
    for i in range(1,4):
        point_dict[i] = [spine1['all_points_x'][i], spine1['all_points_y'][i]]
    for i in range(4,7):
        point_dict[i] = [spine2['all_points_x'][i-4], spine2['all_points_y'][i-4]]
    '''
    point_dict[1] = [spine1['all_points_x'][0], spine1['all_points_y'][0]]
    point_dict[2] = [spine1['all_points_x'][3], spine1['all_points_y'][3]]
    point_dict[3] = [spine1['all_points_x'][2], spine1['all_points_y'][2]]
    point_dict[4] = [spine2['all_points_x'][0], spine2['all_points_y'][0]]
    point_dict[5] = [spine2['all_points_x'][1], spine2['all_points_y'][1]]
    point_dict[6] = [spine2['all_points_x'][-1], spine2['all_points_y'][-1]]    #关键点6个坐标
    #储存成dataFrame的形式
    info = get_angle_and_perpdistanceRatio(point_dict[2],point_dict[3],point_dict[4],point_dict[5])
    res_dic = {}
    res_dic['图片序号'+pic_num+'比例'] = info[1]
    #res_dic['图片序号'+pic_num+'foot_x'] = info[2]
    #res_dic['图片序号'+pic_num+'foot_y'] = info[3]
    ##是否要加4条相关直线的长度与角度？
    slope, length = get_slope_and_length(point_dict[1], point_dict[2])
    res_dic['图片序号'+pic_num+'line1'], res_dic['图片序号'+pic_num+'slope1'] = length, slope
    slope, length = get_slope_and_length(point_dict[2], point_dict[3])
    res_dic['图片序号'+pic_num+'line2'], res_dic['图片序号'+pic_num+'slope2'] = length, slope
    slope, length = get_slope_and_length(point_dict[5], point_dict[4])
    res_dic['图片序号'+pic_num+'line3'], res_dic['图片序号'+pic_num+'slope3'] = length, slope
    slope, length = get_slope_and_length(point_dict[5], point_dict[6])
    res_dic['图片序号'+pic_num+'line4'], res_dic['图片序号'+pic_num+'slope4'] = length, slope
    res_dic['图片序号'+pic_num+'PSD'] = get_PSD(point_dict[1], point_dict[2], point_dict[4])
    return pd.DataFrame.from_dict(res_dic,orient='index')

#通过编号获得3张图片的Info    spine_num:第几块脊柱   regions:脊柱区域
def get_multiInfo(person_num,spine_num,regions):
    spine_num += 1
    pic = {}
    pic[2] = person_num + '_2.jpg'
    pic[3] = person_num + '_3.jpg'
    pic[4] = person_num + '_4.jpg'
    for v in pic.values():
        if v not in regions.keys():
            return 1
    dfs = {}
    for k in range(2,5):
        tmp = get_info(spine_num,str(k), regions[pic[k]])
        if type(tmp) == int: return 0
        dfs[k] = pd.DataFrame(tmp.T)
    for k in range(2,5):
        res = pd.concat(list(dfs.values()), axis=1)
    res['编号'] = person_num
    return res

#根据脊柱编号获得数据 num:脊柱是第num块
def getLabel(num, labels):
    df = pd.DataFrame()
    df['编号'] = range(len(labels))
    df['label'] = range(len(labels))
    i = 0
    if num<=0 or num>=6: return 
    for k,v in labels.items():
        df.iloc[i,0]=k
        df.iloc[i,1] = np.nan
        if v == [] or v[num-1] == -1:
            pass
        else:
            df.iloc[i,1] = v[num-1]
        i += 1
    df.dropna(inplace=True)
    return df

