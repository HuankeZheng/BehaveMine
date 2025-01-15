import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DataLoader(Dataset):
    def __init__(self, data, window_size, input_dim):
        self.data = data
        self.window_size = window_size
        self.act_dim = input_dim[0]

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        data = self.data[idx:idx + self.window_size + 1]

        activity_list = [item['activity_id'] for item in data]
        stage_list = [item['time_stage'] for item in data]

        act_input = torch.tensor(activity_list[:-1], dtype=torch.int)
        stage_input = torch.tensor(stage_list[:-1], dtype=torch.int)

        sw_lists = [item['sensor_happen_id_list'] for item in data]
        sw_lists = sw_lists[:-1]

        sb_lists = [item['sensor_between_id_list'] for item in data]
        sb_lists = sb_lists[:-1]

        act_target = torch.zeros((self.window_size, self.act_dim))
        for i, item in enumerate(activity_list[1:]):
            act_target[i, item] = 1

        return act_input, stage_input, sw_lists, sb_lists, act_target


def collate_fn(batch):
    batch_size = len(batch)
    act_input, stage_input, sw_lists, sb_lists, act_target = zip(*batch)

    sw_lists = [torch.tensor(lists) for batch in sw_lists for lists in batch]
    sw_lens = [len(sensor_list) for sensor_list in sw_lists]
    sw_lens = torch.tensor(sw_lens).view(-1)
    sw_padded = pad_sequence(sw_lists, batch_first=True, padding_value=0)
    sw_padded = sw_padded.view(batch_size, -1, sw_padded.size(1))

    sb_lists = [torch.tensor(lists) for batch in sb_lists for lists in batch]
    sb_lens = [len(sensor_list) for sensor_list in sb_lists]
    sb_lens = torch.tensor(sb_lens).view(-1)
    sb_padded = pad_sequence(sb_lists, batch_first=True, padding_value=0)
    sb_padded = sb_padded.view(batch_size, -1, sb_padded.size(1))

    act_input = torch.stack(act_input)
    stage_input = torch.stack(stage_input)

    act_target = torch.stack(act_target)
    return (act_input, stage_input, sw_padded, sb_padded), act_target, (
        sw_lens, sb_lens)
