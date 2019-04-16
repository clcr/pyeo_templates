#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.ensemble as ens
import gdal
from osgeo import ogr, osr,gdal
import os,errno
from tempfile import TemporaryDirectory
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
import pickle
import csv
import pdb


# In[3]:


#from tpot import TPOTClassifier
#import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
#from tpot.builtins import StackingEstimator


# In[4]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import BernoulliNB


# In[5]:


def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype=None):
    """Creates an empty gdal dataset with the same dimensions, projection and geotransform. Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified"""
    driver = gdal.GetDriverByName(format)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    out_dataset = driver.Create(out_path,
                                xsize=in_dataset.RasterXSize,
                                ysize=in_dataset.RasterYSize,
                                bands=bands,
                                eType=datatype)
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


# In[6]:


def reshape_raster_for_ml(image_array):
    """Reshapes an array from gdal order [band, y, x] to scikit order [x*y, band]"""
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


# In[7]:


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'
    WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong quietly and in a way that'll cause the most issues
     further on down the line."""
    with TemporaryDirectory() as td:
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromEPSG(shape_projection_id)
        image = gdal.Open(image_path)
        image_gt = image.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5]
        ras_path = os.path.join(td, "poly_ras")
        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute=attribute,
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_Int16,
            outputSRS=shape_projection
        )
        # This produces a rasterised geotiff that's right, but not perfectly aligned to pixels.
        # This can probably be fixed.
        gdal.Rasterize(ras_path, shape_path, options=ras_params)
        rasterised_shapefile = gdal.Open(ras_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        local_x, local_y = get_local_top_left(image, rasterised_shapefile)
        shape_sparse = sp.coo_matrix(shape_array)
        y, x, features = sp.find(shape_sparse)
        training_data = np.empty((len(features), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        image_view = image_array[:,
                     local_y: local_y + rasterised_shapefile.RasterYSize,
                     local_x: local_x + rasterised_shapefile.RasterXSize
                     ]
        for index in range(len(features)):
            training_data[index, :] = image_view[:, y[index], x[index]]
    return training_data, features


# In[8]:


def get_local_top_left(raster1, raster2):
    """Gets the top-left corner of raster1 in the array of raster 2; WRITE A TEST FOR THIS"""
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])


# In[9]:


def point_to_pixel_coordinates(raster, point, oob_fail=False):
    """Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the point.
    Point can be an ogr point object, a wkt string or an x, y tuple or list. Assumes north-up non rotated.
    Will floor() decimal output"""
    # Equation is rearrangement of section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html
    if isinstance(point, str):
        point = ogr.CreateGeometryFromWkt(point)
        x_geo = point.GetX()
        y_geo = point.GetY()
    if isinstance(point, list) or isinstance(point, tuple):  # There is a more pythonic way to do this
        x_geo = point[0]
        y_geo = point[1]
    if isinstance(point, ogr.Geometry):
        x_geo = point.GetX()
        y_geo = point.GetY()
    gt = raster.GetGeoTransform()
    x_pixel = int(np.floor((x_geo - floor_to_resolution(gt[0], gt[1])) / gt[1]))
    y_pixel = int(np.floor((y_geo - floor_to_resolution(gt[3], gt[5] * -1)) / gt[5]))  # y resolution is -ve
    return x_pixel, y_pixel


# In[10]:


def floor_to_resolution(input, resolution):
    """Returns input rounded DOWN to the nearest multiple of resolution."""
    return input - (input % resolution)


# In[11]:


def reshape_ml_out_to_raster(classes, width, height):
    """Reshapes an output [x*y] to gdal order [y, x]"""
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array


# In[12]:


def classify_image(in_image_path, model, out_image_path,num_chunks =10):
    print("Classifying image")
    image = gdal.Open(in_image_path)
    image_array = image.GetVirtualMemArray()
    features_to_classify = reshape_raster_for_ml(image_array)
    width = image.RasterXSize
    height = image.RasterYSize
    out_chunks = []
    for i, chunk in enumerate(np.array_split(features_to_classify, num_chunks)):
        print("Classifying {0}".format(i))
        chunk_copy = np.copy(chunk)
        out_chunks.append(model.predict(chunk_copy))
    out_classes = np.concatenate(out_chunks)
    image = gdal.Open(in_image_path)
    out_image = create_matching_dataset(image, out_image_path)
    image_array = None
    image = None
    out_image_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_image_array[...] = reshape_ml_out_to_raster(out_classes, width, height)
    out_image_array = None
    out_image = None


# In[13]:


def save_model(model, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(model, fp)


# In[14]:


def load_model(filepath):
    with open(filepath, "rb") as fp:
        return pickle.load(fp)


# In[15]:


def summarise_training(in_classes,out_csv, sumarise_type = 'count'):
    df = pd.DataFrame(in_classes, columns = ['type'])
    if sumarise_type == 'count':
        training_summary = df.groupby(['type']).size()
    elif sumarise_type == 'mean':
        training_summary = df.groupby(['type']).mean()
    elif sumarise_type == 'median':
        training_summary = df.groupby(['type']).median
    else:
        print('Add more math func here, can only summarise in terms of count, mean or median now')
    training_summary.to_csv(out_csv)


# In[ ]:


def train_model(features,classes, model_format):
    model = model
    model.fit(features, classes)
    scores = cross_val_score(model, features, classes, cv=cross_val_repeats)
    return model, score


# In[ ]:


def train_model_rf(features,classes):
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
    min_samples_split=16, n_estimators=100, n_jobs=-1, class_weight='balanced')
    model.fit(features, classes)
    scores = cross_val_score(model, features, classes, cv=5)
    return model, scores


# In[ ]:


def getallinfo(g):
    (x_min, pixel_width, rotation, y_max, rotation, pixel_height) = g.GetGeoTransform()
    rows = g.RasterYSize
    cols = g.RasterXSize
    bands = g.RasterCount
    x_max = (cols*pixel_width) + x_min
    y_min = y_max + (rows*pixel_height)
    return x_min, pixel_width, rotation, y_max, rotation, pixel_height, rows, cols, bands, x_max, y_min


# In[ ]:


def clipShp(input_raster,shpfile_path,output_fldr):
    (x_min, pixel_width, rotation, y_max, rotation, pixel_height, rows, cols, bands, x_max, y_min) = getallinfo(input_raster)
  
 #   print getallinfo(input_raster)
    clipfile = os.path.join(output_fldr, 'outline_clip.shp' )
    remove_exist(clipfile)
    print ('The clipped shapefile to the extent of the raster, resultant shp is saved in ' + clipfile )
    os.system('ogr2ogr -clipdst ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + clipfile + ' ' + shpfile_path)   
    return clipfile


# In[ ]:


def remove_exist(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


# In[ ]:


def cal_ratio(alldata, classnumber):
    results = []
    results.append(classnumber)    
    newclass = [num for num in alldata if np.logical_or(num == classnumber, classnumber in num)]
    results.append(len(newclass))
    results.append(round(len(newclass)/len(alldata),6))
    
    return results


# In[ ]:


def validate_classes(inRaster, shpdir, field_name='GRID_CODE', out_fldr=' ',nodata = 0):
    s0data = gdal.Open(inRaster)
    with open(os.path.join(out_fldr, os.path.basename(inRaster)[:-4] + '_' + "validation.csv"), 'w') as fs:
        writer = csv.writer(fs)
        writer.writerow(['validateClass', 'PredictedClass', 'number', 'ratio'])
        inshp = shpdir
        print('~validating ... ' + inshp)
        clipfile = clipShp(s0data, inshp, out_fldr)# clip shp to the extent of the raster
        print('rasterise the shapefile')
        (x_min, pixel_width, rotation, y_max, rotation, pixel_height, rows, cols, bands, x_max, y_min) = getallinfo(
            s0data)
        clipd_shp_rst = os.path.join(out_fldr, 'groundata_raster.tif')
        remove_exist(clipd_shp_rst)
        os.system('gdal_rasterize -a_nodata 0 -a ' + field_name + ' -ot Float32 -l ' + os.path.basename(
            clipfile[:-4]) + ' -te ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(
            y_max) + ' -tR ' + str(pixel_width) + ' ' + str(-pixel_height) + ' ' + clipfile + ' ' + clipd_shp_rst)
        r = gdal.Open(clipd_shp_rst)
        shp_array1 = r.GetRasterBand(1).ReadAsArray()
        for class_i in np.unique(shp_array1):
            all_class = np.zeros(shp_array1.shape)
            all_class[shp_array1 == class_i] = 1  # now the shp only have 0 and 1, 1 are where validating points are
            bs_array = s0data.GetRasterBand(1).ReadAsArray()  # .astype(np.float)
            new = all_class* bs_array
            for i in np.unique(new):
                if i == nodata:
                    continue  # skip
                else:
                    stat = []
                    stat.append(str(class_i))
                    testclass = str(i)
                    stat.append(testclass)
                    num = len(new[new == i])
                    stat.append(num)
                    total_nonnum = len(new[new != 0.])
                    ratio = round(float(num) / float(total_nonnum), 6)
                    stat.append(ratio)
                    writer.writerow(stat)
                    print(stat)
            writer.writerow([])
    fs.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




