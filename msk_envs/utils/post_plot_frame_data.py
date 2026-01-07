import json
from .checkpoint_parser import FrameData
from .pdf_log_builder import create_pdf_output


def main():
    with open("deploy_frame_data_0.json", "r") as f:
        frame_data_json = json.load(f)

    frame_data = []
    for frame_json in frame_data_json:
        frame = FrameData.from_dict(frame_json)
        frame_data.append(frame)
    create_pdf_output(frame_data, "deploy_frame_data.pdf")


if __name__ == "__main__":
    main()
