import React, { Component } from "react";
import * as d3 from "d3";
import ReactECharts from "echarts-for-react";
import axios from "axios";

import {
    Button,
    Modal,
    Row,
    Col,
    Typography,
    Radio,
    Empty,
    Select,
    message,
    Tooltip,
    Tag,
    Image,
    Descriptions,
    Card,
    Spin,
} from "antd";

const { Option } = Select;
// 常量设置
const { Title, Text } = Typography;
const xScale = d3.scaleLinear();
const yScale = d3.scaleLinear();
const maxValue = 20; // 缩放大小
const lineWidth = 0.2; // 分割线宽度
const rows = 65; //每行个数
const cols = 80; //每列个数
const imgSize = 12; //图片大小

const mapColors = [
    "#e7f1ff",
    "#b3d2ed",
    "#5ca6d4",
    "#1970ba",
    "#0c3b80",
    "#042950",
];
const group_color = [
    "#6fe214",
    "#2e2b7c",
    "#c94578",
    "#ebe62b",
    "#f69540",
    "#9f9a9a", // back
];
const mapLevel = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"];
const category_name = ["no cancer", "cancer", "high cancer"];
const category_color = ["#40b373", "#d4b446", "#ee8826"]; // and more
const noise_tag_color = ["#90b7b7", "#a9906c", "#ef9a9a"];
const noise_tag_name = ["clean", "noise", "high-noise"];

export default class MapVision extends Component {
    constructor(props) {
        super(props);
        this.state = {
            heatEvent: null,
            heatMapType: "close",
            index: [],
            load: false,
            visible: false,
            gridSize: 1,
            confirmLoading: false,
            submitLoading: false,
            barOption: {},
            areaOption: {},
            selectedPatch: 0,
            tipShowColor: [],
        };
    }


    componentDidMount = async () => {
        this.drawChart();
        this.setState({
            load: false,
        });
    };

    drawChart = async () => {
        const colors = ["#e7f1ff", "#0c3b80"];
        const imgId = 20;
        const This = this;
        xScale.domain([0, rows]).range([0, imgSize * rows]);
        yScale.domain([0, cols]).range([imgSize * cols, 0]);
        d3.select("#map").selectAll("svg").remove();

        // 初始化zoom
        const zoom = d3
            .zoom()
            .scaleExtent([1, maxValue])
            .translateExtent([
                [-5, -5],
                [imgSize * rows, imgSize * cols],
            ])
            .on("zoom", zoomed);
        // 热力图
        var colorScale = d3
            .scaleLinear()
            .domain([0, 1])
            .range(colors)
            .interpolate(d3.interpolateHcl);
        // 初始化画布
        const mainGroup = d3
            .select("#map")
            .append("svg")
            .attr("preserveAspectRatio", "xMinYMin meet")
            .attr("width", "100%")
            .attr("height", "100%");
        //导入数据
        const imgs = mainGroup.selectAll("image").data([0]);
        d3.csv("./data/1.csv", function (e, csvdata) {
            var margin = lineWidth;
            var file = e["0"]
            var x_y = file.split("_");
            var y = parseInt(x_y[1]);
            var x = parseInt(x_y[3].split("png")[0]);
            var p = "./data/init_data_image/image_" + imgId;
            var path = p + "/" + file;
            imgs
                .enter()
                .append("svg:image")
                .attr("xlink:href", path)
                .attr("row", x)
                .attr("col", y)
                .attr("x", imgSize * x +margin)
                .attr("y", imgSize * y +margin)
                .attr("width", imgSize - lineWidth +margin)

        });
        // 添加
        mainGroup.call(zoom);
        drawGrid();
        // drawPatches();
        // 恢复大小
        d3.select("#zoom_out").on("click", () => {
            mainGroup.transition().call(zoom.transform, d3.zoomIdentity, [0, 0]);
        }

        );
        // 绘制网格
        function drawGrid(event) {
            mainGroup.selectAll("line").remove();
            var margin = lineWidth;
            if (event != null && event.transform.k > 25)
                margin = (lineWidth * event.transform.k) / 20;
            var grid = (g) =>
                g
                    .attr("stroke", "blue")
                    .attr("stroke-opacity", 0.5)
                    .attr("stroke-width", margin)
                    .call((g) =>
                        g
                            .append("g")
                            .selectAll("line")
                            .data(xScale.ticks(rows))
                            .join("line")
                            .attr("x1", (d) => xScale(This.state.gridSize * d))
                            .attr("x2", (d) => xScale(This.state.gridSize * d))
                            .attr("y2", imgSize * cols)
                            .attr("transform", event == null ? null : event.transform)
                    )
                    .call((g) =>
                        g
                            .append("g")
                            .selectAll("line")
                            .data(yScale.ticks(cols))
                            .join("line")
                            .attr("y1", (d) => yScale(This.state.gridSize * d))
                            .attr("y2", (d) => yScale(This.state.gridSize * d))
                            .attr("x2", imgSize * rows)
                            .attr("transform", event == null ? null : event.transform)
                    );
            mainGroup.call(grid);
        }


        function zoomed(event) {
            This.setState({
                heatEvent: event,
            });
            drawGrid(event);

        }
    };

    render() {
        return (
            <>

                                <div id="map" style={{height:"80vh"}}>

                                </div>


            </>
        );
    }
}
