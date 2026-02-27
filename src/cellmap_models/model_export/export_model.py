import torch.onnx
import torch
import os

omnx_version = 17


def export_torch_model(model, input_shape, folder_result, metadata=None):
    if not os.path.exists(folder_result):
        os.makedirs(folder_result)
    model.eval()
    print(f"Exporting model to {folder_result}")

    if metadata is not None:
        from .generate_metadata import export_metadata
        export_metadata(metadata)
    pt_file = os.path.join(folder_result, "model.pt")
    onnx_file = os.path.join(folder_result, "model.onnx")
    ts_file = os.path.join(folder_result, "model.ts")
    ep_file = os.path.join(folder_result, "model.pt2")
    try:
        # Export to PyTorch pickle
        torch.save(model, pt_file)
        print(f"Model saved to {pt_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        # Export via torch.export (ExportedProgram)
        dummy_input = torch.rand(input_shape)
        exported = torch.export.export(model, (dummy_input,))
        torch.export.save(exported, ep_file)
        print(f"Model saved to {ep_file}")
    except Exception as e:
        print(f"Error exporting with torch.export: {e}")

    try:
        dummy_input = torch.rand(input_shape)
        scripted_model = torch.jit.trace(model, dummy_input)
        scripted_model.save(ts_file)
        print(f"Model saved to {ts_file}")
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")

    try:
        # Export to ONNX
        try:
            dummy_input = torch.rand(input_shape)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_file,
                export_params=True,
                opset_version=omnx_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
            print(f"Model saved to {onnx_file}")
        except Exception as e:
            try:
                print(f"Error exporting to ONNX: {e}")
                print("trying to export it with external data")
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_file,
                    export_params=True,
                    opset_version=omnx_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    size_threshold=1024
                    )
                print(f"Model saved to {onnx_file}")
            except Exception as e:
                print(f"Error exporting to ONNX with external data: {e}")
                print("will try to make it smaller")
                try:
                    import torch.quantization as tq

                    model_quantized = tq.quantize_dynamic(
                        model,
                        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                        dtype=torch.qint8
                    )
                    torch.onnx.export(
                        model_quantized,
                        dummy_input,
                        onnx_file,
                        export_params=True,
                        opset_version=omnx_version,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        size_threshold=1024
                    )
                    print(f"Model saved to {onnx_file}")
                except Exception as e:
                    print(f"Error exporting to ONNX with quantization: {e}")

    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
