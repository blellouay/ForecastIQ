from src.data.demand_classification import classify_demand,demand_statistics

def test_known_categories_and_origin_safety():
    assert classify_demand([2]*20)=="Smooth"
    assert classify_demand([1,8]*10)=="Erratic"
    assert classify_demand([1,0,1,0,1,0])=="Intermittent"
    assert classify_demand([1,0,9,0,1,0,9,0])=="Lumpy"
    assert classify_demand([0]*10)=="No sales"
def test_cv2_uses_positive_values_only():
    assert demand_statistics([1,0,1,0,1])["CV2"]==0
