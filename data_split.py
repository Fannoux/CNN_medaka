import glob
import os
import shutil
import pandas as pd
import argparse
import tqdm
import sys


# INFO_FILE = '/home/fanny/Work/EBI/MIKK/Ziram/Germany_data_plate_image.csv'
# INFO_FILE = '/nfs/research/birney/users/fanny/medaka/ziram_analysis/PlateWell_to_Image_CO4selected_09062023.csv'

# FOLDER_PICS = './Severity Score Pics/*/*CO4*'
# FOLDER_PICS = '/nfs/ftp/private/indigene_ftp/upload/NCSU/ziram_images/**/**/*CO4*.tif'

# DATASET_PATH = '/home/fanny/Work/EBI/Indigene/Ziram/CNN_pytorch/dataset_cnn'
# DATASET_PATH = '/nfs/research/birney/users/fanny/medaka/ziram_analysis/dataset_cnn'

# COLUMN_SEP = 'kinked'


def open_infofile(file, column_sep, ratio_test=0.2, filename='filename', force=False):
    # dict_folder = {'test':os.path.join(dataset_path, 'test_set'),
    #                'training': os.path.join(dataset_path, 'training_set')}
    df = pd.read_csv(file)
    print('[INFO] Shape of the df from the file', df.shape)
    
    
    # df[filename] = df[filename].str.rsplit('/', n=1, expand=True)[1]
    assert df[filename].value_counts().max() == 1, 'Problem with naming images, repetition or missed value !'


    if ('folder' in df.columns ) & (force == False):
        assert set(df['folder'].unique()) == set(['test_set', 'training_set']), "Folder column values not set as \'test_set\', \'training_set\'"
        # image_random_ls = df.loc[df['folder'] == 'test_set', filename].tolist()

    else: 
        # image_random_ls = []
        df['folder'] = 'training_set'
        for val, n_val in dict(df[column_sep].value_counts()).items():
            print('[INFO] value and number of files required for the test set', val, int(n_val * ratio_test))
            random_index = df[df[column_sep].astype(str) == str(val)].sample(int(n_val * ratio_test)).index
            df.loc[random_index, 'folder'] = 'test_set'
            # image_random_ls  += df[df[ column_sep].astype(str) == str(val)].sample(int(n_val * ratio_test))[filename].tolist()
            # dict_folder[f'training_{str(val)}'] = os.path.join(dataset_path, 'training_set', str(val))
            # dict_folder[f'test_{str(val)}'] = os.path.join(dataset_path, 'test_set', str(val))
    # image_random_ls = [os.path.basename(image).split('.', n=1)[0] for image in image_random_ls]
    print('[INFO] Repartition per sets\n', df['folder'].value_counts(normalize=True))
    assert df['folder'].nunique() == 2, 'Wrong number of set'

    # return df, image_random_ls, dict_folder
    return df


def make_dataset_dirs(dict_folder, dataset_path='dataset'):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for folder in dict_folder.values():
        if not os.path.exists(folder):
            os.mkdir(folder)

def move_file(file_name, destination_folder):
    try:
        shutil.copy2(file_name, destination_folder)
    except:
        print(f'[ERROR] Error while copying {file_name} into {destination_folder}')


def main(folder_pics, info_file, column_sep, dataset_path='dataset', filename='filename', ratio_test=0.2, force=False):
    df = open_infofile(info_file, column_sep, filename=filename, ratio_test=ratio_test, force=force)
    
    # df, random_list, dict_folder = open_infofile(info_file, column_sep, force=force)
    dict_folder = {'test':os.path.join(dataset_path, 'test_set'),
                   'training': os.path.join(dataset_path, 'training_set')}
    
    make_dataset_dirs(dict_folder, dataset_path=dataset_path)
    
    # Move dependent on basename without extension
    test_files_ls = df.loc[df['folder'] == 'test_set', filename].dropna().tolist()
    train_files_ls = df.loc[df['folder'] == 'training_set', filename].dropna().tolist()
    test_files_ls = [im.split('.')[0] for im in test_files_ls]
    train_files_ls = [im.split('.')[0] for im in train_files_ls]
    warn_im = []
    for image_name in tqdm.tqdm(glob.glob(folder_pics), desc='Spliting dataset'):
        image = os.path.basename(image_name).split('.')[0]
        # print(image)
        try:
            # val = df.loc[(df[filename].str.contains(image)), column_sep].tolist()[0]
            # print(image, df.loc[(df[filename]== image), filename].nunique(), val)
            if image in test_files_ls:
                move_file(image_name, dict_folder['test'])
            elif image in train_files_ls:
                move_file(image_name, dict_folder['training'])
            else:
                warn_im.append(image)
       
            # df.loc[(df[filename].str.contains(image)), 'folder'] = dict_folder[destination_folder]
        except:
            print('[ERROR] Problem with finding values for file ', image_name)

    print('[WARN] Spare images without values not included in datasets', len(warn_im))
    df.to_csv(os.path.join(dataset_path, 'Repartition_images_sets.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data split between training and test set", argument_default=argparse.SUPPRESS)
    parser.add_argument("folder_pics", type=str)
    parser.add_argument("info_file", type=str)
    parser.add_argument("column_sep", type=str)
    parser.add_argument("-p", "--dataset_path", type=str, default='dataset')
    parser.add_argument("-r", "--ratio_test", type=float, default=0.2)
    parser.add_argument("-n", "--filename", type=str, default='filename')
    parser.add_argument("-f", "--force", type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))


# todo reduce or convert images ! see image analysis