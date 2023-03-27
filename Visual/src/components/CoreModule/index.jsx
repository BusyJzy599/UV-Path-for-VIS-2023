/*
 * @Date: 2022-04-17 18:26:37
 * @LastEditors: JZY
 * @LastEditTime: 2022-12-17 17:22:12
 * @FilePath: /project/Visual/src/components/CoreModule/index.jsx
 */
import React, { Component } from "react";
import { Row, Col, Card } from "antd";
import ScatterModel from "./ScatterModel";

export default class CoreModule extends Component {
  constructor(props) {
    super(props);
    this.state = {
      choosePatches: props.choosePatches,
    };
  }
  componentDidMount = () => {
    this.props.onChildEvent(this);
  };
  changeDeletePatches = (p) => {
    this.setState({
      choosePatches: p,
    });
    this.scatterModelRef.changeDeletePatches(p);
  };
  changeChoosePatches = async (p) => {
    await this.setState({
      choosePatches: p,
    });
    this.props.changeChoosePatches(this.state.choosePatches);
  };
  handleScatterModuleEvent = (ref) => {
    this.scatterModelRef = ref;
  };
  render() {
    return (
      <>
        <Card bordered={false} hoverable={true}>
          <Row>
            <Col span={24}>
              <ScatterModel
                onChildEvent={this.handleScatterModuleEvent}
                ref={this.scatterModelRef}
                changeChoosePatches={this.changeChoosePatches}
                choosePatches={this.state.choosePatches}
                mapValid={this.props.mapValid}
                sample_Data={this.props.sample_Data}
                WSI_Data={this.props.WSI_Data}
                current_iteration={this.props.current_iteration}
              />
            </Col>
          </Row>
        </Card>
      </>
    );
  }
}
