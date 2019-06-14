import torch

class data_prefetcher():
    def __init__(self,
                 loader,
                 im_data_1, im_info_1, gt_boxes_1, num_boxes_1,
                 im_data_2, im_info_2, gt_boxes_2, num_boxes_2):
        self.im_data_1 = im_data_1
        self.im_info_1 = im_info_1
        self.gt_boxes_1 = gt_boxes_1
        self.num_boxes_1 = num_boxes_1

        self.im_data_2 = im_data_2
        self.im_info_2 = im_info_2
        self.gt_boxes_2 = gt_boxes_2
        self.num_boxes_2 = num_boxes_2

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def process_input(self):
        self.im_data_1.data.resize_(self.next_input_1[0].size()).copy_(self.next_input_1[0])
        self.im_info_1.data.resize_(self.next_input_1[1].size()).copy_(self.next_input_1[1])
        self.gt_boxes_1.data.resize_(self.next_input_1[2].size()).copy_(self.next_input_1[2])
        self.num_boxes_1.data.resize_(self.next_input_1[3].size()).copy_(self.next_input_1[3])

        self.im_data_2.data.resize_(self.next_input_2[0].size()).copy_(self.next_input_2[0])
        self.im_info_2.data.resize_(self.next_input_2[1].size()).copy_(self.next_input_2[1])
        self.gt_boxes_2.data.resize_(self.next_input_2[2].size()).copy_(self.next_input_2[2])
        self.num_boxes_2.data.resize_(self.next_input_2[3].size()).copy_(self.next_input_2[3])

    def preload(self):
        try:
            self.next_input_1, self.next_input_2 = next(self.loader)
        except StopIteration:
            self.next_input_1 = None
            self.next_input_2 = None
            return
        with torch.cuda.stream(self.stream):
            self.process_input()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.im_data_1, self.im_info_1, self.gt_boxes_1, self.num_boxes_1, \
                self.im_data_2, self.im_info_2, self.gt_boxes_2, self.num_boxes_2
        self.preload()
        return input