# Floorplan Representations: Pros, Cons, and Optimization Implications

## Overview

Floorplan representations are crucial in Electronic Design Automation (EDA) as they determine how chip layouts are encoded and optimized. Different representations offer various trade-offs in expressiveness, computational efficiency, and algorithmic compatibility. This document discusses major floorplan representations and their impact on optimization algorithms.

## 1. Sequence Pair (SP) Representation

### How it Works
- Uses two permutations of blocks: `(Γ+, Γ-)`
- Relative positions determined by block order in sequences
- A block X is left of Y if `pos(X, Γ+) < pos(Y, Γ+)` and `pos(X, Γ-) < pos(Y, Γ-)`

### Pros
- **Compact representation**: O(n) space complexity for n blocks
- **Complete coverage**: Can represent all possible floorplan topologies
- **Easy to generate**: Simple to create random valid sequences
- **Natural for evolutionary algorithms**: Easy mutation/crossover operations

### Cons
- **Redundant encodings**: Multiple sequence pairs can represent same floorplan
- **No direct area computation**: Requires O(n²) time to evaluate packing
- **Difficult constraint handling**: Hard to enforce adjacency/whitespace constraints
- **Memory inefficient for large designs**: Quadratic evaluation time becomes bottleneck

### Optimization Algorithm Impact
- **Genetic Algorithms**: Excellent - fast mutation (swap elements) and crossover
- **Simulated Annealing**: Good for local search, but slow evaluation limits scalability
- **RL/Transformers**: Challenging due to redundant representations and slow evaluation

## 2. Polish Expression (PE) / Slicing Tree

### How it Works
- Postfix expression using blocks and operators (+ horizontal, * vertical)
- Example: `A B + C *` means (A+B)*C
- Represents hierarchical slicing structure

### Pros
- **Intuitive visualization**: Tree structure matches designer intuition
- **Guaranteed slicing floorplans**: All represented layouts are slicing-based
- **Compact and unique**: Each floorplan has unique expression
- **Fast evaluation**: O(n) time for area computation

### Cons
- **Limited expressiveness**: Cannot represent all possible non-slicing topologies
- **Fixed hierarchy**: Difficult to modify local regions without affecting global structure
- **Whitespace handling**: Hard to distribute whitespace flexibly
- **Constraint satisfaction**: Challenging to enforce complex placement rules

### Optimization Algorithm Impact
- **Tree search algorithms**: Very effective - natural hierarchical modifications
- **Simulated Annealing**: Good for local perturbations, fast evaluation helps
- **Transformers**: Excellent - can be treated as "language" for sequence modeling
- **RL**: Strong for policy learning, but limited by expressiveness constraints

## 3. B*-Tree Representation

### How it Works
- Balanced binary tree where leaves are blocks, internal nodes are operators
- Each subtree represents a rectangular region
- Maintains topology through tree structure

### Pros
- **Hierarchical structure**: Natural for multi-level optimization
- **Flexible topology**: Can represent both slicing and non-slicing floorplans
- **Efficient updates**: Local changes don't affect entire tree
- **Good for constraints**: Easy to enforce adjacency and shape constraints

### Cons
- **Complex maintenance**: Tree balancing and restructuring operations
- **Memory overhead**: Additional pointers and node information
- **Evaluation complexity**: O(n log n) for area computation
- **Implementation complexity**: More sophisticated data structures needed

### Optimization Algorithm Impact
- **Simulated Annealing**: Excellent - local moves (rotate, swap, delete-insert) are efficient
- **Genetic Algorithms**: Good but complex crossover operations
- **RL**: Very suitable - action space of tree operations is well-defined
- **Transformers**: Challenging - tree structure doesn't map naturally to sequences

## 4. Corner Block List (CBL)

### How it Works
- Records sequence of blocks placed at corners during placement process
- Maintains placement order and corner assignments
- Compact representation of placement sequence

### Pros
- **Very compact**: O(n) space for any number of blocks
- **Fast generation**: Easy to create random placements
- **Natural ordering**: Reflects actual placement process
- **Good for routability**: Captures realistic placement sequences

### Cons
- **Limited topology control**: Cannot represent arbitrary topologies
- **Evaluation overhead**: Still requires O(n²) packing evaluation
- **Constraint handling**: Difficult to enforce specific placement rules
- **Non-intuitive**: Hard for humans to interpret or modify directly

### Optimization Algorithm Impact
- **Simulated Annealing**: Good for placement-style optimization
- **Construction heuristics**: Natural for bottom-up construction algorithms
- **RL**: Suitable for sequential decision making
- **Transformers**: Limited - not as semantically rich as other representations

## 5. Transitive Closure Graph (TCG)

### How it Works
- Graph where nodes are blocks, edges represent adjacency relationships
- Transitive closure captures all implied relationships
- Compact graph representation of block positions

### Pros
- **Direct relationship encoding**: Explicitly represents block adjacencies
- **Flexible constraints**: Easy to add/modify adjacency requirements
- **Scalable representation**: O(n²) worst case but often sparse in practice
- **Good for optimization**: Natural for graph-based algorithms

### Cons
- **Complexity**: O(n³) time for transitive closure computation
- **Storage**: Can be dense for highly connected designs
- **Ambiguity**: Multiple graphs can represent same floorplan
- **Evaluation**: Still requires geometric computation for area/wirelength

### Optimization Algorithm Impact
- **Graph algorithms**: Excellent - natural for network flow, matching algorithms
- **Constraint optimization**: Very strong for handling complex constraints
- **RL**: Challenging due to state space complexity
- **Transformers**: Difficult - graph structure not sequence-based

## Comparative Analysis

### Representation Selection Criteria

| Criteria | Best Choice | Worst Choice |
|----------|-------------|--------------|
| **Expressiveness** | Sequence Pair | Polish Expression |
| **Evaluation Speed** | Polish Expression | Sequence Pair/TCG |
| **Memory Efficiency** | Corner Block List | B*-Tree |
| **Constraint Handling** | TCG/B*-Tree | Sequence Pair |
| **Algorithm Flexibility** | B*-Tree | Corner Block List |
| **Human Interpretability** | Polish Expression | Corner Block List |

### Algorithm-Representation Matching

- **For Fast Optimization**: Polish Expression (fast evaluation) + Simulated Annealing
- **For Global Search**: Sequence Pair + Genetic Algorithms
- **For RL/Transformers**: Polish Expression (language-like) or Sequence Pair (sequential)
- **For Constraint Satisfaction**: B*-Tree or TCG
- **For Hierarchical Design**: B*-Tree

## Emerging Trends

1. **Hybrid Representations**: Combining strengths (e.g., SP-Tree merges sequence pair with tree structure)

2. **Neural Representations**: Learning compressed embeddings that capture floorplan semantics

3. **Multi-resolution Approaches**: Using different representations at different abstraction levels

## Practical Considerations

- **Design Size**: For small designs (<100 blocks), evaluation speed matters less - use most expressive representation
- **Optimization Goals**: Wirelength-focused vs. area-focused vs. constraint-satisfaction affects choice
- **Tool Integration**: Must consider how representation integrates with existing EDA flow
- **Parallelization**: Some representations (sequence pair) parallelize better than others

The choice of representation significantly impacts both the quality of results and computational efficiency. Modern approaches often use hybrid methods or learn optimal representations through machine learning.
