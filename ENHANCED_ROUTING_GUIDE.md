# Enhanced MCP Routing System Guide

## Overview

The enhanced MCP routing system allows your dashboard to intelligently route queries to both **Record Agent** tools and **Agent Workflow** tools based on natural language patterns. This provides seamless access to all functionality across your hierarchical MCP server architecture.

## Architecture

```
Main MCP Server
├── Record Agent (Google Sheets Tools)
│   ├── Check_Holding_Area
│   ├── Check_DO
│   ├── Check_Full_Treatment
│   └── Check_full_ncr
└── Agent Workflow (Complex Tools)
    ├── Admin Functions
    │   ├── generate_do
    │   ├── do_download_link
    │   └── parse_and_log_supplier_delivery_order
    ├── Account Functions
    │   ├── approve_po
    │   ├── partial_po
    │   ├── reject_po
    │   └── get_usd_to_sgd_rate
    ├── Inventory Functions
    │   └── bsl_inventory
    └── AI Functions
        ├── table_format
        ├── analysis
        └── general_query
```

## How It Works

### 1. Intelligent Pattern Matching

The system uses regex patterns to match user queries to appropriate tools:

```python
# Example patterns:
r'(?i)\b(check\s+holding\s+area|list\s+holding\s+area|show\s+holding\s+area|holding\s+area)\b'
r'(?i)\b(approve\s+po|approve\s+purchase\s+order|po\s+approval)\b'
r'(?i)\b(generate\s+do|create\s+do|generate\s+delivery\s+order|create\s+delivery\s+order)\b'
```

### 2. Tool Classification

Each tool is classified by type:

- **record_agent**: Direct Google Sheets access
- **agent_workflow**: Complex tools that go through the Agent sub-workflow

### 3. Argument Extraction

The system automatically extracts relevant arguments from queries:

```python
# Query: "approve po 12345"
# Extracted: {'POnumber': '12345'}

# Query: "partial po 67890 qty 10"  
# Extracted: {'POnumber': '67890', 'partial_qty': '10'}
```

## Usage Examples

### Record Agent Tools

```python
# Holding Area Queries
"check holding area"
"list holding area"
"show holding area items"

# Delivery Order Queries
"check do"
"list delivery orders"
"show delivery order status"

# NCR Queries
"check full ncr"
"list ncr reports"
"show non-conformance reports"

# Treatment Parts
"check full treatment"
"list treatment parts"
```

### Agent Workflow Tools

#### Admin Functions
```python
# Generate DO
"generate new delivery order"
"create do"

# DO Download Links
"get download link for do 12345"
"download do 67890"

# Supplier DO Processing
"parse and log supplier delivery order"
"upload supplier delivery order pdf"
```

#### Account Functions
```python
# PO Approvals
"approve po 12345"
"approve purchase order 67890"

# Partial Approvals
"partial approval for po 12345 qty 10"
"partial po 67890 quantity 5"

# PO Rejections
"reject po 12345"
"reject purchase order 67890"

# Exchange Rates
"get usd to sgd rate"
"current exchange rate"
```

#### Inventory Functions
```python
# Inventory Checks
"check bsl inventory"
"check inventory levels"
"stock status"
```

#### AI Functions
```python
# Table Formatting
"format this data as a table"
"show in table format"

# Analysis
"analyze the previous results"
"provide summary"

# General Queries
"what is the total count?"
"show me overdue items"
```

## Configuration

### 1. Tool Pattern Configuration

Add new tools by extending the `tool_patterns` dictionary in `mcp_client.py`:

```python
self.tool_patterns = {
    # Add new pattern
    r'(?i)\b(your\s+new\s+pattern)\b': {
        'tool': 'Agent',  # or specific tool name
        'type': 'agent_workflow',  # or 'record_agent'
        'sub_function': 'your_function',
        'description': 'Description of what this tool does'
    }
}
```

### 2. Argument Extraction

Extend the `_extract_tool_arguments` method for new tools:

```python
def _extract_tool_arguments(self, message: str, tool_info: Dict) -> Dict:
    arguments = {}
    
    if tool_info.get('sub_function') == 'your_function':
        # Extract your specific arguments
        pattern_match = re.search(r'your_regex_pattern', message, re.IGNORECASE)
        if pattern_match:
            arguments['your_param'] = pattern_match.group(1)
    
    return arguments
```

## Advanced Features

### 1. AI-Enhanced Processing

The system automatically determines when to use AI processing:

- **Agent Workflow Tools**: Always use AI for formatting
- **Record Agent Tools**: Use AI for analysis requests
- **Complex Queries**: Route through AI for better responses

### 2. Context Awareness

The system maintains conversation context for better routing:

- Previous query results
- Conversation history
- Tool interaction patterns

### 3. Fallback Handling

If no specific pattern matches, queries default to the general Agent tool for comprehensive handling.

## Testing

Run the test script to verify routing:

```bash
python test_enhanced_routing.py
```

This will test all tool patterns and show routing decisions.

## Dashboard Integration

The enhanced routing is automatically integrated into the Streamlit dashboard:

- **Categorized Commands**: Commands are displayed by category
- **Tool Information**: Shows which tool will handle each query
- **Enhanced UI**: Better user experience with organized commands

## Benefits

1. **Unified Interface**: Single point of access for all tools
2. **Intelligent Routing**: Automatic tool selection based on query intent
3. **Natural Language**: Users can ask questions naturally
4. **Scalable**: Easy to add new tools and patterns
5. **AI-Enhanced**: Automatic formatting and analysis of responses

## Best Practices

1. **Pattern Specificity**: Make patterns specific enough to avoid conflicts
2. **Argument Validation**: Always validate extracted arguments
3. **Error Handling**: Implement proper fallback for unmatched queries
4. **Testing**: Test new patterns thoroughly before deployment
5. **Documentation**: Document new tools and their usage patterns

## Troubleshooting

### Common Issues

1. **Pattern Conflicts**: Multiple patterns matching the same query
   - Solution: Make patterns more specific

2. **Argument Extraction Fails**: Parameters not extracted correctly
   - Solution: Debug regex patterns and test with sample queries

3. **Wrong Tool Selected**: Query routes to unexpected tool
   - Solution: Check pattern priority and specificity

### Debug Tools

Use the dashboard's debug tools to:
- Discover available server tools
- Test query routing
- View tool information

## Future Enhancements

1. **Machine Learning**: Use ML to improve pattern matching
2. **Dynamic Patterns**: Allow runtime pattern configuration
3. **Performance Optimization**: Cache routing decisions
4. **Advanced Analytics**: Track tool usage patterns
5. **Multi-Language Support**: Support queries in different languages

---

*This enhanced routing system provides a powerful, flexible foundation for accessing all your MCP server tools through natural language queries.* 