"""Loader tests on tiny synthetic CSVs (schema identical to IEEE-CIS)."""

import numpy as np

from fraudops.data.loader import load_joined

TX_HEADER = (
    "TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,card1,card4,"
    "C1,D1,M1,P_emaildomain\n"
)
ID_HEADER = "TransactionID,id_02,id_12,DeviceType,DeviceInfo\n"


def write_csv(path, header: str, rows: list[str]) -> None:
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_join_preserves_all_transactions_and_marks_identity(tmp_path) -> None:
    tx = tmp_path / "train_transaction.csv"
    ident = tmp_path / "train_identity.csv"
    write_csv(
        tx,
        TX_HEADER,
        [
            "1,0,100,50.0,W,100,visa,3.0,1.0,T,gmail.com\n",
            "2,1,200,120.0,H,101,mastercard,2.0,2.0,F,hotmail.com\n",
            "3,0,300,30.0,W,102,visa,1.0,3.0,T,yahoo.com\n",
        ],
    )
    write_csv(
        ident,
        ID_HEADER,
        [
            "1,10.0,Found,desktop,Windows\n",
            "3,30.0,NotFound,mobile,Android\n",
        ],
    )
    df = load_joined(tx, ident)

    assert len(df) == 3  # left join: no transaction row lost
    assert df["TransactionID"].tolist() == [1, 2, 3]
    assert df["has_identity"].tolist() == [True, False, True]
    # identity fields present only where identity existed
    assert df.loc[df.TransactionID == 2, "DeviceInfo"].isna().all()
    assert df.loc[df.TransactionID == 1, "DeviceInfo"].item() == "Windows"
    # dtypes: memory-efficient by design
    assert str(df["TransactionAmt"].dtype) == "float32"
    assert str(df["ProductCD"].dtype) == "category"
    assert str(df["isFraud"].dtype) == "int8"


def test_identity_record_with_nan_deviceinfo_still_flagged(tmp_path) -> None:
    # some real identity records carry no DeviceInfo: the has_identity flag
    # must reflect the presence of an identity record, not of that column
    tx = tmp_path / "train_transaction.csv"
    ident = tmp_path / "train_identity.csv"
    write_csv(
        tx,
        TX_HEADER,
        [
            "1,0,100,50.0,W,100,visa,3.0,1.0,T,gmail.com\n",
        ],
    )
    write_csv(
        ident,
        ID_HEADER,
        [
            "1,10.0,Found,desktop,\n",  # DeviceInfo missing, id_12 present
        ],
    )
    df = load_joined(tx, ident)
    assert bool(df["has_identity"].item())
    assert df["DeviceInfo"].isna().all()


def test_transaction_numeric_nan_stays_nan(tmp_path) -> None:
    tx = tmp_path / "train_transaction.csv"
    ident = tmp_path / "train_identity.csv"
    write_csv(
        tx,
        TX_HEADER,
        [
            "1,0,100,50.0,W,100,visa,,1.0,T,gmail.com\n",
            "2,0,200,60.0,H,101,visa,4.0,,F,\n",
        ],
    )
    write_csv(ident, ID_HEADER, "1,10.0,Found,desktop,Windows\n")
    df = load_joined(tx, ident)
    assert np.isnan(df.loc[df.TransactionID == 1, "C1"].item())
    assert np.isnan(df.loc[df.TransactionID == 2, "D1"].item())
    assert np.isnan(df.loc[df.TransactionID == 2, "P_emaildomain"].item())
