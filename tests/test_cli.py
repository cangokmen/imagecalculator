from PIL import Image

from imagecalculator.cli import main


def test_cli_roundtrip(tmp_path, smooth64, capsys):
    src = tmp_path / "in.png"
    smooth64.save(src)
    icf = tmp_path / "out.icf"
    assert main(["compress", str(src), "--keep", "25%", "-q", "5", "--block-size", "8", "--image-size", "64", "-o", str(icf)]) == 0
    out = capsys.readouterr().out
    assert "keep: 4/15" in out and "tokens:" in out and "bytes over the air" in out
    png = tmp_path / "out.png"
    assert main(["decompress", str(icf), "-o", str(png)]) == 0
    assert Image.open(png).size == (32, 32)
    assert main(["decompress", str(icf), "--size", "full", "-o", str(png)]) == 0
    assert Image.open(png).size == (64, 64)
    assert main(["info", str(icf), "--original", str(src), "--json"]) == 0
    assert '"over_the_air"' in capsys.readouterr().out
    assert main(["compress", str(src), "--keep", "6", "-q", "[[0,3,1],[3,15,10]]", "--block-size", "8", "--image-size", "64", "-o", str(icf)]) == 0
