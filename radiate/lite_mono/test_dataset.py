from datasets.radiate_mono_dataset import RadiateMonoDataset
from datasets.radiate_mono_radar_dataset import RadiateMonoRdarDataset


if __name__ == '__main__':
    
    # -------------------test for the mono dataset----------------------------------
    # train_dataset = RadiateMonoDataset(data_path='../data/radiate_f', mode='train_all', 
    #                                    height=320, width=640,
    #                                    frame_idxs=[0, -1, 1],
    #                                    num_scales=3)
    # print(len(train_dataset))
    # print(train_dataset[0])
    
    # val_dataset = RadiateMonoDataset(data_path='../data/radiate_f', mode='test', 
    #                                    height=320, width=640,
    #                                    frame_idxs=[0, -1, 1],
    #                                    load_depth=True,
    #                                    num_scales=3)
    # print(val_dataset[0])
    
    
    
    #------------------test for the mono radar dataset, train------------------
    train_dataset = RadiateMonoRdarDataset(data_path='../data/radiate_f', mode='train_all', 
                                       height=320, width=640,
                                       frame_idxs=[0, -1, 1],
                                       num_scales=3,
                                       load_unc=True)
    print(len(train_dataset))
    print(train_dataset[0])  
    print(train_dataset[1])  
    print(train_dataset[2])  
    print(train_dataset[3])  
    print(train_dataset[4])  
    
    
    
    # -------------------test for the mono radar dataset----------------------------------
    # val_dataset = RadiateMonoRdarDataset(data_path='../data/radiate_f', mode='test', 
    #                                    height=320, width=640,
    #                                    frame_idxs=[0, -1, 1],
    #                                    load_depth=True,
    #                                    num_scales=3)
    # print(len(val_dataset))
    # print(val_dataset[0])