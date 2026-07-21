import torch

class StringTensorUtils:
    def __init__(self, max_length: int, padding_char: str = '<PAD>'):
        self.max_length = max_length
        self.padding_char = padding_char
        self.char_to_index = {chr(i): i - 32 for i in range(32, 127)}  # ASCII可打印字符
        self.char_to_index[padding_char] = len(self.char_to_index)  # 添加填充字符
        self.index_to_char = {i - 32: chr(i) for i in range(32, 127)}
        self.index_to_char[len(self.char_to_index) - 1] = padding_char  # 添加填充字符的反向映射

    def str_to_tensor(self, input_str: str) -> torch.Tensor:
        indices = [self.char_to_index.get(char, self.char_to_index[self.padding_char]) for char in input_str]
        if len(indices) < self.max_length:
            indices += [self.char_to_index[self.padding_char]] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_str(self, tensor: torch.Tensor) -> str:
        indices = tensor.tolist()
        indices = [index for index in indices if index != self.char_to_index[self.padding_char]]
        return ''.join(self.index_to_char.get(index, '') for index in indices)

if __name__ == "__main__":
    utils = StringTensorUtils(max_length=1000)

    input_str = r"left: A white truck with a tank on the back is driving down the road.\ncenter: A street with a crosswalk and trees in the background.\nright: A traffic light with a red hand signal on a street corner.\n"
    tensor = utils.str_to_tensor(input_str)
    print("String to Tensor:", tensor)

    output_str = utils.tensor_to_str(tensor)
    print("Tensor to String:", output_str)

    # 测试填充
    input_str2 = "Hi"
    tensor2 = utils.str_to_tensor(input_str2)
    print("String to Tensor (with padding):", tensor2)
    output_str2 = utils.tensor_to_str(tensor2)
    print("Tensor to String (with padding):", output_str2)
