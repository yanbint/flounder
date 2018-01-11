import cv2
import sys
import os

def data_clean(img_file, model_type):
    model_list = []
    img = cv2.imread(img_file)
    rows, cols, channels = img.shape
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)                     
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if model_type == "HSV":
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for r in range(rows):
        for c in range(cols):        
            # get values from rgb color space
            R = imgRGB.item(r,c,0)
            G = imgRGB.item(r,c,1)
            B = imgRGB.item(r,c,2)        
            # get values from ycbcr color space   
            Y =  imgYcc.item(r,c,0)
            Cr = imgYcc.item(r,c,1)
            Cb = imgYcc.item(r,c,2)                                                                                                                                        
            if model_type == "HSV":
                H = imgHSV.item(r,c,0)
                S = imgHSV.item(r,c,1)
                V = imgHSV.item(r,c,2)   

            # skin color detection          
            if R > G and R > B:
                if (G >= B and 5 * R - 12 * G + 7 * B >= 0) or (G < B and 5 * R + 7 * G - 12 * B >= 0):
                    if Cr > 135 and Cr < 180 and Cb > 85 and Cb < 135 and Y > 80:
                        if model_type == "RGB":
                            model_list.append([R, G, B])
                        elif model_type == "YCrCb":
                            model_list.append([Y, Cr, Cb])
                        elif model_type == "HSV":   
                            model_list.append([H, S, V])
    return model_list

def data_clean_to_csv(img_file, model_type, clean_file = ''):
    model_list = data_clean(img_file, model_type)
    if clean_file == '':
        clean_file = os.path.dirname(img_file) + os.sep + os.path.basename(img_file).split('.')[0] + '_' + model_type + '.csv'
    with open(clean_file, 'w') as fout:
        for model in model_list:
            fout.write("%d;%d;%d\n" % (model[0], model[1], model[2]))
    
    print "generate model csv: %s" % clean_file
    return clean_file

if __name__=="__main__":
    Usage = "Usage"
    if len(sys.argv) != 3:
        print Usage 
        sys.exit(0)

    model_type = sys.argv[1]
    image_path = sys.argv[2]

    if model_type not in ["RGB", "YCrCb", "HSV"]:
        print Usage
        sys.exit(0)

    data_clean_to_csv(image_path, model_type)