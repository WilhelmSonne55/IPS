#ifndef PARSER_H
#define PARSER_H

//base demosaic algorithm
class Parser
{
   public:
   Parser();
   
    virtual void demosaic();

};

//different algorithm
class ParserImpl : public Parser
{
   public:
    ParserImpl();

    //tips: override make sure not using base function
    void demosaic (void) override;
};

#endif 