
from resnet import * 
import pandas as pd
import torch.utils.data as tdata
from collections import OrderedDict
import pytorch_lightning as pl
from data_loader import * 
from collections import Counter
import argparse

def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        n_key = key.split("model.")[1]
        new_state_dict[n_key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def infer_labels_for_models(model, ground_truth_file_path, data_dir, checkpoint_dir, device, num_workers, batch_size):
    image_name_prediction_mapping = {}
    country_names = ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22']

    for country_name in country_names:
        for urban_flag in [0,1]:
            dataset_helper = DatasetHelper(csv_file = ground_truth_file_path, root_dir = data_dir, 
                  country_name = country_name, urban = urban_flag)
            test_image_paths, _, idx_to_class, class_weights = dataset_helper.get_image_paths(1)
            dataset = WildsDataset(test_image_paths, idx_to_class, output_filenames=True)
            loader = DataLoader(dataset, num_workers = num_workers, batch_size=batch_size, shuffle=False)
            for batch in loader:
                for checkpoint_filename in os.listdir(checkpoint_dir):
                    checkpoint_path = checkpoint_dir + "/" + checkpoint_filename
                    model = load_model(model, checkpoint_path, device)
                    image = batch[0]
                    label = batch[1]
                    imagenames = batch[2]
                    output = model(image)
                    logits = output.softmax(dim=1)
                    preds = logits.argmax(dim=1)
                    for i, imagename in enumerate(imagenames):
                        if imagename not in image_name_prediction_mapping.keys():
                            image_name_prediction_mapping[imagename] = []
                        image_name_prediction_mapping[imagename].append(preds[i].tolist())
                    
    return image_name_prediction_mapping
                
def get_final_preds_df(image_name_prediction_mapping, majority_threshold):
    final_predictions = {}
    for image in image_name_prediction_mapping.keys():
        preds = image_name_prediction_mapping[image]
        if sum(preds) > (majority_threshold*len(preds)):
            final_predictions[image] = 1
        elif sum(preds) < ((1-majority_threshold)*len(preds)):
            final_predictions[image] = -1
        else:
            final_predictions[image] = 0
    df = pd.DataFrame(final_predictions, index=[0]).T
    df["filename"] = df.index
    df = df.rename(columns={0:"label"}).reset_index(drop=True)
    return df

def compute_score(student_data, ground_truth_data, country_number, a):
    country = country_number
    a = -4
    student_data = student_data.sort_values(by=['filename'])
    ground_truth_data = ground_truth_data.sort_values(by=['filename'])
    if country_number != None:
        test_set = ground_truth_data[ground_truth_data.country == country]["label"]
        comp_set = student_data[student_data.country == country]["label"]
    else:
        test_set = ground_truth_data["label"]
        comp_set = student_data["label"]
    correct = (test_set == comp_set).sum()
    incorrect = (((test_set != comp_set) & (comp_set != 0))*-a).sum()
    score = correct + incorrect
    return score/len(comp_set)
    
    

def setupParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--majority_threshold", '-m', type = float, default = 0.7)
    parser.add_argument("--a", '-a', type = int, default = -4)
    parser.add_argument("--num_workers", '-w', type = int, default = 0)
    parser.add_argument("--device", '-de', type = str, default = 'cpu')
    parser.add_argument("--results_dir", '-rd', type = str, default = './')
    parser.add_argument("--batch_size", '-b', type = int, default = 8)
    parser.add_argument("--ground_truth_file_path", '-gt', type = str, default = "../tables/test_table.csv")
    parser.add_argument("--checkpoint_dir", '-checkp', type = str, default = './checkpoints/')
    parser.add_argument("--data_dir", '-d', type = str, default = '/datasets/cs255-sp22-a00-public/tmp_for_partitioned_images3/partitioned_images3/test')
    return parser

def eval_per_country(student_data, ground_truth_data, a):
    for country_number in student_data.country.unique():
        score = compute_score(student_data, ground_truth_data, country_number, a)
        print(f"Score for country {country_number}: {score}")

if __name__ == "__main__":
    parser = setupParser()
    args = parser.parse_args()

    majority_threshold = args.majority_threshold
    num_workers = args.num_workers
    batch_size = args.batch_size
    ground_truth_file_path = args.ground_truth_file_path
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    device = args.device
    results_dir = args.results_dir
    a = args.a
    model = ResNet18(num_classes = 2, num_channels = 8)
    
    image_name_prediction_mapping = infer_labels_for_models(model, ground_truth_file_path, data_dir, checkpoint_dir, device, num_workers, batch_size)
    preds_df = get_final_preds_df(image_name_prediction_mapping, majority_threshold)
    preds_df.to_csv(results_dir + "/predictions.csv")
    
    
    ground_truth_data = pd.read_csv(ground_truth_file_path)
    
    preds_df = preds_df.merge(ground_truth_data[['filename', 'country']], left_on="filename", right_on="filename")
    
    #eval_per_country(preds_df, ground_truth_data, a)




