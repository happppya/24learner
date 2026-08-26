use std::io::{BufRead, BufWriter, Write};

use serde::Deserialize;

use core24::ops::{Action, BinaryOp, UnaryOp};
use core24::solver::{plan_solves, solve_plan};
use core24::state::State;
use core24::value::Elem;

#[derive(Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    elems: Vec<(f64, u8)>,
    #[serde(default)]
    target: f64,
    #[serde(default)]
    lambda: f64,
    #[serde(default)]
    action: Option<ActionDto>,
    #[serde(default)]
    budget: u64,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum ActionDto {
    Binary { i: usize, j: usize, op: String },
    Unary { i: usize, op: String },
}

fn build_state(pairs: &[(f64, u8)], target: f64) -> State {
    State {
        elems: pairs
            .iter()
            .map(|(v, d)| Elem {
                value: *v,
                depth: *d,
            })
            .collect(),
        target,
        steps: 0,
    }
}

fn decode_action(dto: &ActionDto) -> Option<Action> {
    match dto {
        ActionDto::Binary { i, j, op } => {
            let bin = match op.as_str() {
                "add" => BinaryOp::Add,
                "sub" => BinaryOp::Sub,
                "mul" => BinaryOp::Mul,
                "div" => BinaryOp::Div,
                "pow" => BinaryOp::Pow,
                _ => return None,
            };
            Some(Action::Binary {
                i: *i,
                j: *j,
                op: bin,
            })
        }
        ActionDto::Unary { i, op } => {
            let un = match op.as_str() {
                "neg" => UnaryOp::Neg,
                "sqrt" => UnaryOp::Sqrt,
                "ln" => UnaryOp::Ln,
                _ => return None,
            };
            Some(Action::Unary { i: *i, op: un })
        }
    }
}

fn encode_action(action: &Action) -> String {
    match *action {
        Action::Binary { i, j, op } => {
            let name = match op {
                BinaryOp::Add => "add",
                BinaryOp::Sub => "sub",
                BinaryOp::Mul => "mul",
                BinaryOp::Div => "div",
                BinaryOp::Pow => "pow",
            };
            format!("{{\"kind\":\"binary\",\"i\":{i},\"j\":{j},\"op\":\"{name}\"}}")
        }
        Action::Unary { i, op } => {
            let name = match op {
                UnaryOp::Neg => "neg",
                UnaryOp::Sqrt => "sqrt",
                UnaryOp::Ln => "ln",
            };
            format!("{{\"kind\":\"unary\",\"i\":{i},\"op\":\"{name}\"}}")
        }
    }
}

fn applied_response(state: &State, action: &Action) -> String {
    match state.apply(action) {
        Some(next) => {
            let idx = match *action {
                Action::Binary { i, j, .. } => i.min(j),
                Action::Unary { i, .. } => i,
            };
            let elem = &next.elems[idx];
            serde_json::json!({ "ok": true, "elem": [elem.value, elem.depth] }).to_string()
        }
        None => "{\"ok\":false}".to_string(),
    }
}

fn handle(req: &Request) -> String {
    match req.cmd.as_str() {
        "actions" => {
            let state = build_state(&req.elems, req.target);
            let list = state
                .legal_actions()
                .iter()
                .map(encode_action)
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"actions\":[{list}]}}")
        }
        "apply" => {
            let dto = match &req.action {
                Some(a) => a,
                None => return "{\"error\":\"missing action\"}".to_string(),
            };
            let len = req.elems.len();
            let out_of_range = match dto {
                ActionDto::Binary { i, j, .. } => *i >= len || *j >= len || i == j,
                ActionDto::Unary { i, .. } => *i >= len,
            };
            if out_of_range {
                return "{\"error\":\"action index out of range\"}".to_string();
            }
            match decode_action(dto) {
                Some(action) => applied_response(&build_state(&req.elems, req.target), &action),
                None => "{\"error\":\"unknown operator\"}".to_string(),
            }
        }
        "reward" => {
            let state = build_state(&req.elems, req.target);
            serde_json::json!({ "ok": true, "reward": state.shaped_reward(req.lambda) }).to_string()
        }
        "solve" => {
            let start = build_state(&req.elems, req.target);
            match solve_plan(&start, req.budget) {
                Some(plan) => {
                    let verified = plan_solves(&start, &plan);
                    let list = plan.iter().map(encode_action).collect::<Vec<_>>().join(",");
                    format!("{{\"solvable\":true,\"verified\":{verified},\"plan\":[{list}]}}")
                }
                None => "{\"solvable\":false}".to_string(),
            }
        }
        other => format!("{{\"error\":\"unknown cmd {other}\"}}"),
    }
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle(&req),
            Err(err) => {
                serde_json::json!({ "error": format!("malformed request: {err}") }).to_string()
            }
        };
        if writeln!(out, "{response}").is_err() {
            break;
        }
        let _ = out.flush();
    }
}
