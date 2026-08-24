# Intelligent Document Processing using Databricks

An end-to-end Intelligent Document Processing (IDP) pipeline built using Databricks AI functions and Delta Lake.

This project processes PDF documents from a landing folder, understands their content using AI, classifies them into document types, extracts structured business information, stores the processed data in Delta tables, and finally moves successfully processed files to a processed folder.

---

## Project Overview

Organizations often receive large numbers of documents such as:

- Invoices
- Purchase Orders
- Receipts
- Other documents

These documents are usually unstructured PDF files.

The goal of this project is to automatically process these documents and convert their unstructured content into structured data that can be used for analytics and downstream applications.

The project uses Databricks AI functions to perform document parsing, classification, and information extraction.

---

## Architecture

```text
                         PDF Documents
                              |
                              v
                    +-------------------+
                    |   Landing Folder   |
                    +-------------------+
                              |
                              v
                    Binary File Reading
                              |
                              v
                  ai_parse_document()
                              |
                              v
                    Parsed Document
                              |
                              v
                      ai_classify()
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           Invoice      Purchase Order      Receipt
              |               |               |
              v               v               v
        ai_extract()    ai_extract()    ai_extract()
              |               |               |
              v               v               v
       Structured Data Structured Data Structured Data
              |               |               |
              v               v               v
        Delta Silver    Delta Silver    Delta Silver
           Table            Table            Table
              |               |               |
              +---------------+---------------+
                              |
                              v
                     Processed Folder
