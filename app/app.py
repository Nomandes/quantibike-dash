from dash import Dash, html, dcc
import dash
import os

external_stylesheets = ['https://bootswatch.com/5/flatly/bootstrap.min.css']
app = Dash(__name__,use_pages=True,external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Quantibike - Analysis Dashboards'),

    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']} - {page['path']}", href=page["relative_path"]
                )
            )
            for page in dash.page_registry.values()
        ]
    ),

    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)