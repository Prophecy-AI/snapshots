import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

class ChristmasTree:
    def __init__(self, center=(0, 0), angle=0):
        self.center = center
        self.angle = angle
        tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
        tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
        x=np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2],np.float64)
        y=np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1],np.float64)
        self.vertices = list(zip(x, y))
        self.poly = Polygon(self.vertices)
        self.update_poly()

    def update_poly(self):
        self.poly = Polygon(self.vertices)
        self.poly = rotate(self.poly, self.angle, origin=(0, 0), use_radians=False)
        self.poly = translate(self.poly, self.center[0], self.center[1])

    def get_poly(self):
        return self.poly

def parse_value(v):
    if isinstance(v, str) and v.startswith('s'):
        return float(v[1:])
    return float(v)

def score_submission(path):
    df = pd.read_csv(path)
    df['x'] = df['x'].apply(parse_value)
    df['y'] = df['y'].apply(parse_value)
    df['deg'] = df['deg'].apply(parse_value)
    df['N'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    total_score = 0
    for n in sorted(df['N'].unique()):
        group = df[df['N'] == n]
        trees = []
        for _, row in group.iterrows():
            t = ChristmasTree((row['x'], row['y']), row['deg'])
            trees.append(t)
            
        if not trees: continue
        
        min_x = min([t.get_poly().bounds[0] for t in trees])
        max_x = max([t.get_poly().bounds[2] for t in trees])
        min_y = min([t.get_poly().bounds[1] for t in trees])
        max_y = max([t.get_poly().bounds[3] for t in trees])
        
        side = max(max_x - min_x, max_y - min_y)
        total_score += (side**2) / n
        
    return total_score

print(f"Score: {score_submission('/home/submission/submission.csv')}")
