from typing import List, Dict

import cv2
import numpy as np
import torch
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.human import Human
from nobos_commons.data_structures.humans_metadata.algorithm_output_buffer import AlgorithmOutputBuffer
from nobos_commons.data_structures.humans_metadata.algorithm_output_buffer_entry import AlgorithmOutputBufferEntry
from nobos_commons.feature_preparations.feature_vec_producers.from_skeleton_joints.feature_vec_producer_ehpi import \
    FeatureVecProducerEhpi
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import RemoveJointsOutsideImgEhpi, NormalizeEhpi
from torch.autograd import Variable

i = 0
class ActionRecNetEhpi(object):
    def __init__(self, model, feature_vec_producer: FeatureVecProducerEhpi, image_size: ImageSize):
        self.model = model
        self.feature_vec_producer = feature_vec_producer
        self.action_buffer: AlgorithmOutputBuffer = AlgorithmOutputBuffer(buffer_size=32)
        self.remove = RemoveJointsOutsideImgEhpi(image_size)
        self.normalize = NormalizeEhpi(image_size)
        model.cuda()
        model.eval()

    def get_actions(self, humans: List[Human], frame_nr: int) -> Dict[str, np.ndarray]:
        ehpi_vecs = []
        img_save_vecs =  []
        img_save = np.zeros((32, 15, 3), dtype=np.float32) 
        
        for human in humans:
            ehpi_vecs.append(
                AlgorithmOutputBufferEntry(human.uid, self.feature_vec_producer.get_feature_vec(human.skeleton)))
        self.action_buffer.add(ehpi_vecs, frame_nr)
        
        '''
        for human in humans : 
            feature_vector = self.feature_vec_producer.get_feature_vec(human.skeleton)
            print("feature vector :")
            print(feature_vector)
            if i % 31 == 0:
                img_save_vecs.append(feature_vector)
                i = i+1
            print(len(img_save_vecs))
            print(img_save_vecs)
        '''        
        
        
        

        humans_for_action_rec = self.action_buffer.get_all(only_full_buffer=True)
        outputs: Dict[str, np.ndarray] = {}
        for human_id, action_vecs in humans_for_action_rec.items():
            ehpi_img = np.zeros((32, 15, 3), dtype=np.float32)
            for frame_num, action_vec in enumerate(action_vecs):
                if action_vec is None:
                    continue
                ehpi_img[frame_num] = action_vec
                '''
                if frame_num == 31 :
                    ehpi_img_save = np.zeros((32, 15, 3), dtype=np.float32)
                    ehpi_img_save = ehpi_img
                    # print("matrix of perfect version :")
                    # print(ehpi_img_save)
                    ehpi_img_save_transpose = np.transpose(ehpi_img_save, (1, 0, 2))
                    ehpi_img_save_transpose_reshaped =  ehpi_img_save_transpose.reshape(1,1440)
                    # print("reshapedd and perfect version :")
                    # print(*ehpi_img_save_transpose_reshaped)
                   # with open('/Users/sk82620/Desktop/read_ntu_rgbd/deneme1.csv','a') as fd:
                    #    np.savetxt(fd, ehpi_img_save_transpose_reshaped,delimiter=',',fmt='%1.3f' )                   
                '''   
                        
            ehpi_img = np.transpose(ehpi_img, (2, 0, 1))
            # Set Blue Channel to zero
            ehpi_img[2, :, :] = 0
            ehpi_img_not_normalized = ehpi_img
            ehpi_img_not_normalized_array = ehpi_img_not_normalized[1,:,:]
            # print(ehpi_img_not_normalized_array)
            # Normalize EHPI
            tmp_dict = {'x': ehpi_img}
            tmp_dict['x'] = self.remove(tmp_dict)['x']
            
            # print(ehpi_img) # each new column will be added to the end of the matrix
            
            ehpi_img = self.normalize(tmp_dict)['x']
            ehpi_img_normalized_array = ehpi_img[1,:,:]
            # print(ehpi_img_normalized_array)

            # action_img = np.transpose(np.copy(ehpi_img), (2, 1, 0))
            # action_img *= 255
            # action_img = action_img.astype(np.uint8)
            # # action_img = cv2.resize(action_img, (action_img.shape[1] * 30, action_img.shape[0] * 30), cv2.INTER_NEAREST)
            # action_img = cv2.cvtColor(action_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("ehpi", action_img)
            # cv2.waitKey(1)
            # cv2.imwrite(os.path.join(get_create_path("/media/disks/beta/dump/itsc_2019_imgs/ehpi"),
            #                          "{}.png".format(str(frame_nr).zfill(5))), action_img)
            net_input = np.zeros((1, 3, 32, 15), dtype=np.float32)
            net_input_not_normalized = np.zeros((1, 3, 32, 15), dtype=np.float32) 
            net_input[0] = ehpi_img
            net_input_not_normalized[0] = ehpi_img_not_normalized
            
            # print("net input :")
            # print(net_input[0])
            
            '''
            ehpi_img_transpose = np.transpose(ehpi_img, (2, 1, 0))
            print("adding columns at the end of the matrix rehaped perfect :")
            print((ehpi_img_transpose.reshape(1,1440)))
            ehpi_img_transpose_reshaped = ehpi_img_transpose.reshape(1,1440)
            '''
            


            input_seq = Variable(torch.tensor(net_input, dtype=torch.float)).cuda()
            tag_scores = self.model(input_seq).data.cpu().numpy()[0]
            outputs[human_id] = tag_scores
        return outputs 

