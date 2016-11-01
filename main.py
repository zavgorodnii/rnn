# -*- coding: utf-8 -*-


from python.gru import GRUModel, CharSequenceInput


def run_char_gru():
    num_epochs = 500
    num_layers = 2
    truncate_at = 256
    state_size = 1024
    batch_size = 256
    checkpoint = "GorborodskyModel.tf"
    file_name = 'data/gorborodsky.txt'
    with open(file_name, 'r') as f:
        raw_data = f.read()
        raw_data = raw_data.decode("utf-8").lower()
    data = CharSequenceInput(
        raw_data, num_epochs=num_epochs,
        truncate_at=truncate_at, batch_size=batch_size)
    train_model = GRUModel(
        state_size=state_size, num_classes=data.vocab_size,
        truncate_at=truncate_at, num_layers=num_layers, batch_size=batch_size)
    train_model.train(data, checkpoint=checkpoint)
    gen_model = GRUModel(
        state_size=state_size, num_classes=data.vocab_size,
        batch_size=1, truncate_at=1, num_layers=num_layers)
    gen_model.generate_characters(data, num_chars=500, checkpoint=checkpoint)


if __name__ == "__main__":
    run_char_gru()
