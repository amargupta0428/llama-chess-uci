"""Offline tests for move parsing (eval/common.py). No engine, no GPU, no API keys —
these lock in the parsing rules the whole eval depends on: first-token extraction,
move-number/annotation stripping, and hard legality checking in both formats.
A model emitting an illegal or garbage move must come back as None (a measured
outcome), never as an exception or a silently accepted move.
"""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from common import parse_move, parse_move_uci


def test_san_plain_move():
    board = chess.Board()
    assert parse_move("e4", board) == board.parse_san("e4")


def test_san_strips_move_number_and_annotations():
    board = chess.Board()
    assert parse_move("1. e4", board) == board.parse_san("e4")
    assert parse_move("1.e4", board) == board.parse_san("e4")
    assert parse_move("e4!?", board) == board.parse_san("e4")


def test_san_illegal_or_garbage_is_none():
    board = chess.Board()
    assert parse_move("Ke2", board) is None  # illegal from the start position
    assert parse_move("banana", board) is None
    assert parse_move("", board) is None


def test_uci_legal_move():
    board = chess.Board()
    assert parse_move_uci("e2e4", board) == chess.Move.from_uci("e2e4")
    assert parse_move("e2e4", board, fmt="uci") == chess.Move.from_uci("e2e4")


def test_uci_illegal_or_malformed_is_none():
    board = chess.Board()
    assert parse_move_uci("e2e5", board) is None  # well-formed but illegal
    assert parse_move_uci("zz9x", board) is None
    assert parse_move_uci("", board) is None


def test_uci_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    assert parse_move_uci("a7a8q", board) == chess.Move.from_uci("a7a8q")


def test_uci_takes_first_token_only():
    board = chess.Board()
    assert parse_move_uci("e2e4 e7e5", board) == chess.Move.from_uci("e2e4")
